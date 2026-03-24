import pytest
import torch
from dataclasses import dataclass
import sys
import importlib

from classes.config import Config
from classes.checkpoint_manager import JambaCheckpointManager


# --- Fixtures ---


@pytest.fixture
def mock_config(mocker):
    """Provide a minimal Config instance without triggering file IO."""
    mocker.patch.object(Config, "load_homophones")
    cfg = Config()
    # Minimize layers and experts to keep test tensors small
    cfg.jamba_config.num_hidden_layers = 1
    cfg.jamba_config.num_experts = 2
    return cfg


# --- Test fuse_experts ---


@dataclass
class FuseExpertsCase:
    name: str
    state_dict_setup: str
    expected_fused_count: int


FUSE_EXPERTS_CASES = [
    FuseExpertsCase(
        name="fuses_unfused_layers",
        state_dict_setup="unfused",
        expected_fused_count=1,
    ),
    FuseExpertsCase(
        name="skips_missing_or_already_fused_layers",
        state_dict_setup="empty",
        expected_fused_count=0,
    ),
]


@pytest.mark.parametrize("case", FUSE_EXPERTS_CASES, ids=lambda c: c.name)
def test_fuse_experts(mock_config, case: FuseExpertsCase):
    manager = JambaCheckpointManager(mock_config)
    state_dict = {}

    # Setup the dummy state dictionary
    if case.state_dict_setup == "unfused":
        base_key = "model.layers.0.feed_forward.experts"
        for i in range(2):
            state_dict[f"{base_key}.{i}.gate_proj.weight"] = torch.ones(2, 2)
            state_dict[f"{base_key}.{i}.up_proj.weight"] = torch.ones(2, 2)
            state_dict[f"{base_key}.{i}.down_proj.weight"] = torch.ones(2, 2)

    new_dict, fused_count = manager.fuse_experts(state_dict)

    assert fused_count == case.expected_fused_count

    if case.expected_fused_count > 0:
        base_key = "model.layers.0.feed_forward.experts"
        # Assert fused keys were created
        assert f"{base_key}.gate_up_proj" in new_dict
        assert f"{base_key}.down_proj" in new_dict
        # Assert original keys were deleted
        assert f"{base_key}.0.gate_proj.weight" not in new_dict


# --- Test prepare_for_fast_path & _process_file ---


@dataclass
class EarlyReturnCase:
    name: str
    create_dir: bool


@pytest.mark.parametrize(
    "case",
    [
        EarlyReturnCase("no_dir", False),
        EarlyReturnCase("empty_dir", True),
    ],
    ids=lambda c: c.name,
)
def test_prepare_for_fast_path_early_return(mocker, tmp_path, mock_config, case: EarlyReturnCase):
    """Test scenarios where no checkpoint processing should occur."""
    target_dir = tmp_path / "ckpt"
    if case.create_dir:
        target_dir.mkdir()

    manager = JambaCheckpointManager(mock_config)
    mock_fuse = mocker.patch.object(manager, "fuse_experts")

    manager.prepare_for_fast_path(target_dir)

    mock_fuse.assert_not_called()


@dataclass
class ProcessCase:
    name: str
    fused_count: int
    expect_save: bool


@pytest.mark.parametrize(
    "case",
    [
        ProcessCase("needs_fusing", 1, True),
        ProcessCase("already_fused", 0, False),
    ],
    ids=lambda c: c.name,
)
def test_prepare_for_fast_path_safetensors(mocker, tmp_path, mock_config, case: ProcessCase):
    """Test the safetensors file processing path, including whether it saves."""
    target_dir = tmp_path / "ckpt"
    target_dir.mkdir()
    (target_dir / "model.safetensors").touch()

    mocker.patch("classes.checkpoint_manager.HAS_SAFETENSORS", True)
    mock_load = mocker.patch("classes.checkpoint_manager.load_file", return_value={"mock": "data"})
    mock_save = mocker.patch("classes.checkpoint_manager.save_file")

    manager = JambaCheckpointManager(mock_config)
    mocker.patch.object(manager, "fuse_experts", return_value=({"mock": "fused"}, case.fused_count))

    manager.prepare_for_fast_path(target_dir)

    mock_load.assert_called_once()
    if case.expect_save:
        mock_save.assert_called_once()
    else:
        mock_save.assert_not_called()


@dataclass
class PytorchFallbackCase:
    name: str
    has_safetensors_lib: bool
    create_safetensors_file: bool


@pytest.mark.parametrize(
    "case",
    [
        PytorchFallbackCase("pytorch_only", True, False),
        PytorchFallbackCase("safetensors_file_but_no_lib", False, True),
    ],
    ids=lambda c: c.name,
)
def test_prepare_for_fast_path_pytorch(mocker, tmp_path, mock_config, case: PytorchFallbackCase):
    """Test the fallback to pytorch_model.bin when safetensors is unavailable."""
    target_dir = tmp_path / "ckpt"
    target_dir.mkdir()
    (target_dir / "pytorch_model.bin").touch()

    if case.create_safetensors_file:
        (target_dir / "model.safetensors").touch()

    mocker.patch("classes.checkpoint_manager.HAS_SAFETENSORS", case.has_safetensors_lib)
    mock_load = mocker.patch("torch.load", return_value={"mock": "data"})
    mock_save = mocker.patch("torch.save")

    manager = JambaCheckpointManager(mock_config)
    mocker.patch.object(manager, "fuse_experts", return_value=({"mock": "fused"}, 1))

    manager.prepare_for_fast_path(target_dir)

    mock_load.assert_called_once()
    mock_save.assert_called_once()


def test_safetensors_import_error(mocker):
    """Force an ImportError on the module level to ensure 100% line coverage."""
    # 1. Mask 'safetensors.torch' in sys.modules so Python thinks it doesn't exist
    mocker.patch.dict(sys.modules, {"safetensors.torch": None})

    # 2. Force Python to re-execute the module-level code
    import classes.checkpoint_manager

    importlib.reload(classes.checkpoint_manager)

    # 3. Assert our except block caught it and set the flag correctly
    assert classes.checkpoint_manager.HAS_SAFETENSORS is False

    # 4. Cleanup: The mocker will restore sys.modules automatically,
    # but we need to reload the module one last time so we don't break other tests!
    mocker.stopall()
    importlib.reload(classes.checkpoint_manager)
