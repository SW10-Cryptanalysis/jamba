import pytest
import sys
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import MagicMock

from classes.config import Config
from classes.trainer import JambaTrainingPipeline


@pytest.fixture
def mock_config(mocker):
    mocker.patch.object(Config, "load_homophones")
    cfg = Config()
    cfg.output_dir = Path("/mock/output")
    cfg.data_dir = Path("/mock")
    return cfg


@pytest.fixture
def mock_pipeline(mock_config):
    """Provides an uninitialized pipeline instance to test methods in isolation."""
    pipeline = object.__new__(JambaTrainingPipeline)
    pipeline.cfg = mock_config
    return pipeline


# --- Test Initialization ---


def test_pipeline_init(mocker, mock_config):
    """Test that the __init__ method correctly chains the setup functions."""
    mock_verify = mocker.patch.object(JambaTrainingPipeline, "_verify_transformers_version")
    mock_inject = mocker.patch.object(JambaTrainingPipeline, "_inject_mamba_kernels")
    mock_init_model = mocker.patch.object(JambaTrainingPipeline, "_initialize_model", return_value="mock_model")
    mock_setup_trainer = mocker.patch.object(JambaTrainingPipeline, "_setup_trainer", return_value="mock_trainer")
    mock_ckpt_mgr = mocker.patch("classes.trainer.JambaCheckpointManager")

    pipeline = JambaTrainingPipeline(mock_config)

    assert pipeline.cfg == mock_config
    mock_verify.assert_called_once()
    mock_inject.assert_called_once()
    mock_init_model.assert_called_once()
    mock_setup_trainer.assert_called_once()
    mock_ckpt_mgr.assert_called_once_with(mock_config)

    assert pipeline.model == "mock_model"
    assert pipeline.trainer == "mock_trainer"
    assert pipeline.checkpoint_manager == mock_ckpt_mgr.return_value


# --- Test Version Verification ---


@dataclass
class VersionCheckCase:
    name: str
    version: str
    expect_error: bool


@pytest.mark.parametrize(
    "case",
    [
        VersionCheckCase("valid_version", "5.3.0", False),
        VersionCheckCase("invalid_version", "5.2.0", True),
    ],
    ids=lambda c: c.name,
)
def test_verify_transformers_version(mocker, mock_pipeline, case: VersionCheckCase):
    mocker.patch("transformers.__version__", case.version)

    if case.expect_error:
        with pytest.raises(RuntimeError, match=f"Requires v5.3.x, found {case.version}"):
            mock_pipeline._verify_transformers_version()
    else:
        mock_pipeline._verify_transformers_version()  # Should not raise


# --- Test Mamba Kernel Injection ---


@dataclass
class KernelInjectCase:
    name: str
    setup_type: str


@pytest.mark.parametrize(
    "case",
    [
        KernelInjectCase("direct_import", "direct"),
        KernelInjectCase("fallback_import", "fallback"),
        KernelInjectCase("import_error", "fail"),
    ],
    ids=lambda c: c.name,
)
def test_inject_mamba_kernels(mocker, mock_pipeline, case: KernelInjectCase):
    mock_mamba = mocker.Mock()
    mock_causal = mocker.Mock()
    mock_jamba = mocker.Mock()
    mock_ssu = mocker.Mock()
    mock_ssd = mocker.Mock()

    # Build parent mocks to prevent Python's import system from generating detached child mocks
    mock_transformers = mocker.Mock()
    mock_transformers.models.jamba.modeling_jamba = mock_jamba

    mock_mamba_parent = mocker.Mock()
    mock_mamba_parent.ops.selective_scan_interface = mock_mamba
    mock_mamba_parent.ops.triton.selective_state_update = mock_ssu
    mock_mamba_parent.ops.triton.ssd_combined = mock_ssd

    modules = {
        "transformers": mock_transformers,
        "transformers.models": mock_transformers.models,
        "transformers.models.jamba": mock_transformers.models.jamba,
        "transformers.models.jamba.modeling_jamba": mock_jamba,
        "mamba_ssm": mock_mamba_parent,
        "mamba_ssm.ops": mock_mamba_parent.ops,
        "mamba_ssm.ops.selective_scan_interface": mock_mamba,
        "mamba_ssm.ops.triton": mock_mamba_parent.ops.triton,
        "mamba_ssm.ops.triton.selective_state_update": mock_ssu,
        "mamba_ssm.ops.triton.ssd_combined": mock_ssd,
        "causal_conv1d": mock_causal,
    }

    # Simulate ImportError by removing from sys.modules AND deleting attributes
    if case.setup_type == "fail":
        modules["mamba_ssm.ops.selective_scan_interface"] = None
        del mock_mamba_parent.ops.selective_scan_interface
    elif case.setup_type == "fallback":
        modules["mamba_ssm.ops.triton.selective_state_update"] = None
        del mock_mamba_parent.ops.triton.selective_state_update

    mocker.patch.dict(sys.modules, modules)
    mock_logger = mocker.patch("classes.trainer.logger")

    mock_pipeline._inject_mamba_kernels()

    if case.setup_type == "fail":
        mock_logger.error.assert_called_once()
        assert "Kernel Bridge failed" in mock_logger.error.call_args[0][0]
    else:
        mock_logger.info.assert_called_once_with("MAMBA 2.3.0 DETECTED: Successfully bridged kernels for Jamba.")
        assert mock_jamba.mamba_inner_fn == mock_mamba.mamba_inner_fn
        assert mock_jamba.is_fast_path_available is True

        if case.setup_type == "direct":
            assert mock_jamba.selective_state_update == mock_ssu.selective_state_update
        else:
            assert mock_jamba.selective_state_update == mock_ssd.mamba_split_conv_open_loop_scan_combined


# --- Test Model and Trainer Setup ---


def test_initialize_model(mocker, mock_pipeline):
    mock_jamba_config = mocker.patch("classes.trainer.JambaConfig")
    mock_jamba_lm = mocker.patch("classes.trainer.JambaForCausalLM")

    model = mock_pipeline._initialize_model()

    mock_jamba_config.assert_called_once()
    mock_jamba_lm.assert_called_once_with(mock_jamba_config.return_value)
    mock_jamba_lm.return_value.bfloat16.assert_called_once()
    assert model == mock_jamba_lm.return_value.bfloat16.return_value


def test_setup_trainer(mocker, mock_pipeline):
    mock_args = mocker.patch("classes.trainer.TrainingArguments")
    mock_ds = mocker.patch("classes.trainer.PretokenizedCipherDataset")
    mock_collator = mocker.patch("classes.trainer.CipherDataCollator")
    mock_trainer = mocker.patch("classes.trainer.Trainer")

    mock_pipeline.model = "mock_model"
    trainer = mock_pipeline._setup_trainer()

    mock_args.assert_called_once()
    assert mock_ds.call_count == 2
    mock_collator.assert_called_once_with(mock_pipeline.cfg)
    mock_trainer.assert_called_once_with(
        model="mock_model",
        args=mock_args.return_value,
        train_dataset=mock_ds.return_value,
        eval_dataset=mock_ds.return_value,
        data_collator=mock_collator.return_value,
    )
    assert trainer == mock_trainer.return_value


# --- Test Run Execution Loop ---


@dataclass
class RunCase:
    name: str
    last_ckpt: str | None
    is_zero_process: bool


@pytest.mark.parametrize(
    "case",
    [
        RunCase("resume_and_save", "mock/path/ckpt-100", True),
        RunCase("resume_no_save", "mock/path/ckpt-100", False),
        RunCase("scratch_and_save", None, True),
    ],
    ids=lambda c: c.name,
)
def test_run(mocker, mock_pipeline, case: RunCase):
    mock_get_last_ckpt = mocker.patch("classes.trainer.get_last_checkpoint", return_value=case.last_ckpt)
    mock_logger = mocker.patch("classes.trainer.logger")

    mock_pipeline.trainer = MagicMock()
    mock_pipeline.trainer.is_world_process_zero.return_value = case.is_zero_process
    mock_pipeline.checkpoint_manager = MagicMock()

    mock_pipeline.run()

    mock_get_last_ckpt.assert_called_once_with(str(mock_pipeline.cfg.output_dir))

    if case.last_ckpt:
        mock_pipeline.checkpoint_manager.prepare_for_fast_path.assert_called_once_with(str(mock_pipeline.cfg.output_dir))
        mock_logger.info.assert_any_call(f"Found checkpoint: {case.last_ckpt}. Resuming...")
    else:
        mock_pipeline.checkpoint_manager.prepare_for_fast_path.assert_not_called()
        mock_logger.warning.assert_called_once_with("No checkpoint found. Starting training from scratch.")

    mock_pipeline.trainer.train.assert_called_once_with(resume_from_checkpoint=case.last_ckpt)

    if case.is_zero_process:
        # Note: os.path.join resolves differently based on OS, so we verify exact arg passed
        import os

        expected_save_path = os.path.join(str(mock_pipeline.cfg.output_dir), "final_model")
        mock_pipeline.trainer.save_model.assert_called_once_with(expected_save_path)
    else:
        mock_pipeline.trainer.save_model.assert_not_called()
