import pytest
import sys
import os
from pathlib import Path
from dataclasses import dataclass

from classes.config import TRANSFORMER_VERSION, Config
from classes.trainer import JambaTrainingPipeline


@pytest.fixture
def mock_config(mocker):
    """Provides a mocked Config object for the pipeline."""
    mocker.patch.object(Config, "load_homophones")
    mocker.patch.object(Config, "output_dir", new_callable=mocker.PropertyMock, return_value=Path("/mock/output"))
    mocker.patch.object(Config, "training_dir", new_callable=mocker.PropertyMock, return_value=Path("/mock/train"))
    mocker.patch.object(Config, "validation_dir", new_callable=mocker.PropertyMock, return_value=Path("/mock/val"))

    cfg = Config()
    cfg.data_dir = Path("/mock")
    cfg.use_spaces = False
    cfg.pad_token_id = 0
    return cfg


@pytest.fixture
def mock_pipeline(mock_config):
    """Provides an uninitialized pipeline instance to test methods in isolation."""
    pipeline = object.__new__(JambaTrainingPipeline)
    pipeline.cfg = mock_config
    return pipeline


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


@dataclass
class VersionCheckCase:
    """Test cases for version verification."""

    name: str
    version: str
    expect_error: bool


@pytest.mark.parametrize(
    "case",
    [
        VersionCheckCase("valid_version", f"{TRANSFORMER_VERSION}.0", False),
        VersionCheckCase("invalid_version", "5.2.0", True),
    ],
    ids=lambda c: c.name,
)
def test_verify_transformers_version(mocker, mock_pipeline, case: VersionCheckCase):
    """Tests that the transformers version check behaves as expected."""
    mocker.patch("transformers.__version__", case.version)

    if case.expect_error:
        with pytest.raises(RuntimeError, match=f"Requires v{TRANSFORMER_VERSION}.x, found {case.version}"):
            mock_pipeline._verify_transformers_version()
    else:
        mock_pipeline._verify_transformers_version()


@dataclass
class KernelInjectCase:
    """Test cases for kernel injection logic."""

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
    """Tests the Mamba kernel injection bridging logic and fallback behaviors."""
    mock_mamba = mocker.Mock()
    mock_causal = mocker.Mock()
    mock_jamba = mocker.Mock()
    mock_ssu = mocker.Mock()
    mock_ssd = mocker.Mock()

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


def test_initialize_model(mocker, mock_pipeline):
    """Tests the initialization and configuration of the Jamba model."""
    mock_jamba_config = mocker.patch("classes.trainer.JambaConfig")
    mock_jamba_lm = mocker.patch("classes.trainer.JambaForCausalLM")

    model = mock_pipeline._initialize_model()

    mock_jamba_config.assert_called_once()
    mock_jamba_lm.assert_called_once_with(mock_jamba_config.return_value, attn_implementation="flash_attention_2")
    mock_jamba_lm.return_value.bfloat16.assert_called_once()
    assert model == mock_jamba_lm.return_value.bfloat16.return_value


def test_setup_trainer(mocker, mock_pipeline):
    """Tests the setup of the Hugging Face Trainer and data collator."""
    mock_args = mocker.patch("classes.trainer.TrainingArguments")
    mock_ds = mocker.patch("classes.trainer.PretokenizedCipherDataset")
    mock_collator = mocker.patch("classes.trainer.CipherDataCollator")
    mock_trainer = mocker.patch("classes.trainer.Trainer")

    mock_pipeline.model = "mock_model"
    trainer = mock_pipeline._setup_trainer()

    mock_args.assert_called_once()
    assert mock_ds.call_count == 2
    mock_ds.assert_any_call(mock_pipeline.cfg.training_dir, mock_pipeline.cfg)
    mock_ds.assert_any_call(mock_pipeline.cfg.validation_dir, mock_pipeline.cfg)

    mock_collator.assert_called_once_with(mock_pipeline.cfg.pad_token_id)

    mock_trainer.assert_called_once_with(
        model="mock_model",
        args=mock_args.return_value,
        train_dataset=mock_ds.return_value,
        eval_dataset=mock_ds.return_value,
        data_collator=mock_collator.return_value,
    )
    assert trainer == mock_trainer.return_value


@dataclass
class RunCase:
    """Test cases for the main training execution loop."""

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
    """Tests the execution run, verifying checkpoint loading and model saving."""
    mock_get_last_ckpt = mocker.patch("classes.trainer.get_last_checkpoint", return_value=case.last_ckpt)
    mock_logger = mocker.patch("classes.trainer.logger")

    mock_pipeline.trainer = mocker.Mock()
    mock_pipeline.trainer.is_world_process_zero.return_value = case.is_zero_process
    mock_pipeline.checkpoint_manager = mocker.Mock()

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
        expected_save_path = os.path.join(str(mock_pipeline.cfg.output_dir), "final_model")
        mock_pipeline.trainer.save_model.assert_called_once_with(expected_save_path)
    else:
        mock_pipeline.trainer.save_model.assert_not_called()
