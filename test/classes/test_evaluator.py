import pytest
import torch
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import MagicMock

from classes.config import Config
from classes.evaluator import JambaEvaluator


@pytest.fixture
def mock_config(mocker):
    """Provide a minimal Config instance without triggering file IO."""
    mocker.patch.object(Config, "load_homophones")
    cfg = Config()
    cfg.output_dir = Path("/mock/output")
    cfg.data_dir = Path("/mock")
    cfg.batch_size = 2
    cfg.dataloader_num_workers = 1
    return cfg


# --- Test Initialization and Model Loading ---


@dataclass
class InitLoadCase:
    name: str
    file_exists: bool
    provided_path: Path | None
    cuda_available: bool


INIT_LOAD_CASES = [
    InitLoadCase("default_path_exists_cpu", True, None, False),
    InitLoadCase("custom_path_exists_cuda", True, Path("/custom/model"), True),
    InitLoadCase("path_missing", False, None, False),
]


@pytest.mark.parametrize("case", INIT_LOAD_CASES, ids=lambda c: c.name)
def test_evaluator_init_and_load_model(mocker, mock_config, case: InitLoadCase):
    """Test device selection, path resolution, and model loading/exceptions."""
    # Arrange mocks
    mocker.patch("torch.cuda.is_available", return_value=case.cuda_available)
    mocker.patch("pathlib.Path.exists", return_value=case.file_exists)

    # Mock the class at the import location to bypass Hugging Face's PyTorch checks
    mock_jamba_class = mocker.patch("classes.evaluator.JambaForCausalLM")
    mocker.patch("classes.evaluator.JambaEvaluator._prepare_dataloader", return_value="dummy_loader")

    expected_path = case.provided_path or (mock_config.output_dir / "final_model")

    # Act & Assert
    if not case.file_exists:
        with pytest.raises(FileNotFoundError, match="Could not find model"):
            JambaEvaluator(mock_config, model_path=case.provided_path)
    else:
        evaluator = JambaEvaluator(mock_config, model_path=case.provided_path)

        # Assert Device selection
        expected_device = "cuda" if case.cuda_available else "cpu"
        assert evaluator.device.type == expected_device

        # Assert Path resolution and loader initialization
        assert evaluator.model_path == expected_path
        assert evaluator.val_loader == "dummy_loader"

        # Assert Model loading
        mock_jamba_class.from_pretrained.assert_called_once_with(
            expected_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        evaluator.model.eval.assert_called_once() # type: ignore


# --- Test DataLoader Preparation ---


def test_prepare_dataloader(mocker, mock_config):
    """Ensure the dataloader is built with correct dataset and collator args."""
    # Prevent heavy __init__ logic
    mocker.patch("classes.evaluator.JambaEvaluator._load_model")

    mock_dataset = mocker.patch("classes.evaluator.PretokenizedCipherDataset")
    mock_collator = mocker.patch("classes.evaluator.CipherDataCollator")
    mock_dataloader = mocker.patch("classes.evaluator.DataLoader", return_value="loader_instance")

    evaluator = JambaEvaluator(mock_config)

    assert evaluator.val_loader == "loader_instance"
    mock_dataset.assert_called_once_with(mock_config.validation_dir, mock_config)
    mock_collator.assert_called_once_with(mock_config)
    mock_dataloader.assert_called_once_with(
        mock_dataset.return_value,
        batch_size=mock_config.batch_size,
        collate_fn=mock_collator.return_value,
        num_workers=mock_config.dataloader_num_workers,
    )


# --- Test Evaluation Batch Processing ---


@dataclass
class ProcessBatchCase:
    name: str
    extract_samples: bool
    logits: torch.Tensor
    labels: torch.Tensor
    expected_errors: int
    expected_symbols: int
    expected_samples_len: int


PROCESS_BATCH_CASES = [
    ProcessBatchCase(
        name="perfect_match_no_extraction",
        extract_samples=False,
        # Logits shape: (batch_size=1, seq_len=3, vocab_size=4) -> Preds: [0, 1, 2]
        logits=torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]),
        labels=torch.tensor([[0, 1, 2]]),
        expected_errors=0,
        expected_symbols=3,
        expected_samples_len=0,
    ),
    ProcessBatchCase(
        name="mixed_errors_with_masking",
        extract_samples=True,
        # Logits -> Preds: Batch 0: [0, 1, 0], Batch 1: [1, 0, 1]
        logits=torch.tensor(
            [
                [[1, 0], [0, 1], [1, 0]],
                [[0, 1], [1, 0], [0, 1]],
            ]
        ),
        # Labels -> Batch 0: [0, 0, ignored], Batch 1: [1, 1, 1]
        labels=torch.tensor(
            [
                [0, 0, -100],  # 1 error on 2nd element
                [1, 1, 1],  # 1 error on 2nd element
            ]
        ),
        expected_errors=2,
        expected_symbols=5,
        expected_samples_len=2,
    ),
    ProcessBatchCase(
        name="all_masked_out",
        extract_samples=True,
        logits=torch.tensor([[[1, 0]]]),
        labels=torch.tensor([[-100]]),
        expected_errors=0,
        expected_symbols=0,
        expected_samples_len=1,
    ),
    ProcessBatchCase(
        name="cap_extracted_samples_at_three",
        extract_samples=True,
        # Creating a batch size of 4 to ensure min(len(input_ids), 3) is hit
        logits=torch.zeros((4, 2, 2)),
        labels=torch.zeros((4, 2), dtype=torch.long),
        expected_errors=0,
        expected_symbols=8,
        expected_samples_len=3,
    ),
]


@pytest.mark.parametrize("case", PROCESS_BATCH_CASES, ids=lambda c: c.name)
def test_process_evaluation_batch(mocker, mock_config, case: ProcessBatchCase):
    """Test evaluation logic: metric calculation, padding mask handling, and extraction limits."""
    mocker.patch("classes.evaluator.JambaEvaluator._load_model")
    mocker.patch("classes.evaluator.JambaEvaluator._prepare_dataloader")

    evaluator = JambaEvaluator(mock_config)

    # Mock Model Forward Pass
    mock_outputs = MagicMock()
    mock_outputs.logits = case.logits
    evaluator.model = MagicMock(return_value=mock_outputs)

    batch = {
        "input_ids": torch.zeros_like(case.labels),  # Dummy input IDs
        "labels": case.labels,
    }

    errors, symbols, samples = evaluator._process_evaluation_batch(batch, extract_samples=case.extract_samples)

    assert errors == case.expected_errors
    assert symbols == case.expected_symbols
    assert len(samples) == case.expected_samples_len

    if case.extract_samples and case.expected_samples_len > 0:
        assert "truth" in samples[0]
        assert "pred" in samples[0]


# --- Test Master Evaluation Loop ---


@dataclass
class EvaluateLoopCase:
    name: str
    batch_returns: list[tuple[int, int, list]]
    expected_ser: float


EVALUATE_LOOP_CASES = [
    EvaluateLoopCase(
        name="standard_evaluation_loop",
        batch_returns=[
            (2, 10, [{"truth": [1], "pred": [0]}]),  # Batch 0 (extracts sample)
            (3, 10, []),  # Batch 1 (no samples)
        ],
        expected_ser=0.25,  # 5 errors / 20 symbols
    ),
    EvaluateLoopCase(
        name="zero_symbols_protects_div_by_zero",
        batch_returns=[
            (0, 0, []),
        ],
        expected_ser=0.0,
    ),
]


@pytest.mark.parametrize("case", EVALUATE_LOOP_CASES, ids=lambda c: c.name)
def test_evaluate_loop(mocker, mock_config, case: EvaluateLoopCase):
    """Test the full evaluation loop, SER aggregation, and logging outputs."""
    mocker.patch("classes.evaluator.JambaEvaluator._load_model")
    mocker.patch("classes.evaluator.JambaEvaluator._prepare_dataloader")

    evaluator = JambaEvaluator(mock_config)

    # Mock the dataloader to yield exactly the number of batches we parameterized
    evaluator.val_loader = [f"batch_{i}" for i in range(len(case.batch_returns))]  # type: ignore

    # Mock the internal processor to return our parameterized tuples
    mock_process = mocker.patch.object(
        evaluator,
        "_process_evaluation_batch",
        side_effect=case.batch_returns,
    )

    # Mock logger to ensure no print errors and cover logging paths
    mock_logger = mocker.patch("classes.evaluator.logger")

    final_ser = evaluator.evaluate()

    assert final_ser == case.expected_ser

    # Assert correct parameters were passed to the processor using exact kwargs
    if len(case.batch_returns) > 0:
        mock_process.assert_any_call(batch="batch_0", extract_samples=True)
        if len(case.batch_returns) > 1:
            mock_process.assert_any_call(batch="batch_1", extract_samples=False)

    # Ensure final SER was logged correctly
    mock_logger.info.assert_any_call(f"FINAL VALIDATION SER: {final_ser:.6f}")
