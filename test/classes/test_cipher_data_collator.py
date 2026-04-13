import pytest
import torch
from unittest.mock import MagicMock

# Adjust this import based on your actual project structure
from classes import CipherDataCollator


@pytest.fixture
def mock_config() -> MagicMock:
    """Provides a mocked Config object for the collator."""
    config = MagicMock()
    config.max_context = 10
    config.bos_token_id = 1
    config.eos_token_id = 2
    config.space_token_id = 3
    config.pad_token_id = 0
    return config


@pytest.fixture
def collator(mock_config: MagicMock) -> "CipherDataCollator":
    """Instantiates the collator with the mocked config."""
    return CipherDataCollator(mock_config)


def test_empty_batch(collator: "CipherDataCollator") -> None:
    """Ensures an empty batch returns empty tensors gracefully."""
    result = collator([])

    assert result["input_ids"].shape == (0, 0)
    assert result["labels"].shape == (0, 0)
    assert result["attention_mask"].shape == (0, 0)


@pytest.mark.parametrize(
    "max_context, seq_len, expected_len",
    [
        (5, 10, 5),      # Sequence exceeds max_context -> gets truncated
        (None, 10, 10),  # max_context is None -> no truncation
        (15, 10, 10),    # Sequence under max_context -> no truncation
    ],
    ids=["truncates_to_max", "none_context_no_truncation", "under_max_no_truncation"]
)
def test_truncation(
    mock_config: MagicMock, 
    max_context: int | None, 
    seq_len: int, 
    expected_len: int
) -> None:
    """Tests that the collator respects the max_context limit."""
    mock_config.max_context = max_context
    test_collator = CipherDataCollator(mock_config)

    batch = [
        {
            "input_ids": list(range(10, 10 + seq_len)),
            "labels": list(range(10, 10 + seq_len))
        }
    ]

    result = test_collator(batch)

    assert result["input_ids"].shape[1] == expected_len
    assert result["labels"].shape[1] == expected_len


@pytest.mark.parametrize(
    "batch, expected_input_ids, expected_labels, expected_attention_mask",
    [
        # Case 1: Standard padding (pad_token_id = 0, ignore_index = -100)
        (
            [
                {"input_ids": [10, 11], "labels": [10, 11]},
                {"input_ids": [10, 11, 12, 13], "labels": [10, 11, 12, 13]},
            ],
            [[10, 11, 0, 0], [10, 11, 12, 13]], 
            [[10, 11, -100, -100], [10, 11, 12, 13]],
            [[1, 1, 0, 0], [1, 1, 1, 1]]
        ),
        # Case 2: Masking special filler tokens (bos=1, eos=2, space=3)
        (
            [
                {"input_ids": [1, 10, 3, 11, 2], "labels": [1, 10, 3, 11, 2]},
            ],
            [[1, 10, 3, 11, 2]],
            [[-100, 10, -100, 11, -100]],  # 1, 3, and 2 should be replaced with -100
            [[1, 1, 1, 1, 1]]
        ),
        # Case 3: Mixed lengths and special tokens combined
        (
            [
                {"input_ids": [1, 10, 2], "labels": [1, 10, 2]},
                {"input_ids": [10, 11], "labels": [10, 11]},
            ],
            [[1, 10, 2], [10, 11, 0]],
            [[-100, 10, -100], [10, 11, -100]], 
            [[1, 1, 1], [1, 1, 0]]
        )
    ],
    ids=["standard_padding", "filler_token_masking", "mixed_lengths_and_tokens"]
)
def test_padding_and_masking(
    collator: "CipherDataCollator", 
    batch: list[dict[str, list[int]]], 
    expected_input_ids: list[list[int]], 
    expected_labels: list[list[int]], 
    expected_attention_mask: list[list[int]]
) -> None:
    """Tests the core functionality of sequence padding and special token masking."""
    result = collator(batch)

    # Assert tensor equality
    assert torch.equal(result["input_ids"], torch.tensor(expected_input_ids, dtype=torch.long))
    assert torch.equal(result["labels"], torch.tensor(expected_labels, dtype=torch.long))
    assert torch.equal(result["attention_mask"], torch.tensor(expected_attention_mask, dtype=torch.long))
