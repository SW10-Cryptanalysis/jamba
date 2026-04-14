import pytest
import torch
from dataclasses import dataclass

# Adjust this import based on your actual project structure
from classes import CipherDataCollator


def test_empty_batch() -> None:
    """Ensures an empty batch returns empty tensors gracefully."""
    collator = CipherDataCollator()
    result = collator([])

    assert result["input_ids"].shape == (0, 0)
    assert result["labels"].shape == (0, 0)
    assert result["attention_mask"].shape == (0, 0)


@pytest.mark.parametrize(
    "max_context, seq_len, expected_len",
    [
        (5, 10, 5),  # Sequence exceeds max_context -> gets truncated
        (None, 10, 10),  # max_context is None -> no truncation
        (15, 10, 10),  # Sequence under max_context -> no truncation
    ],
    ids=["truncates_to_max", "none_context_no_truncation", "under_max_no_truncation"],
)
def test_truncation(max_context: int | None, seq_len: int, expected_len: int) -> None:
    """Tests that the collator respects the max_context limit."""
    test_collator = CipherDataCollator(0, max_context)

    batch = [{"input_ids": list(range(10, 10 + seq_len)), "labels": list(range(10, 10 + seq_len))}]

    result = test_collator(batch)

    assert result["input_ids"].shape[1] == expected_len
    assert result["labels"].shape[1] == expected_len


@dataclass
class PaddingAndMaskingTestCase:
    """Data for testing the padding and masking logic."""

    name: str
    batch: list[dict[str, list[int]]]
    expected_input_ids: list[list[int]]
    expected_labels: list[list[int]]
    expected_attention_mask: list[list[int]]


PADDING_AND_MASKING_TEST_CASES = [
    PaddingAndMaskingTestCase(
        name="standard_padding",
        batch=[
            {"input_ids": [10, 11], "labels": [10, 11]},
            {"input_ids": [10, 11, 12, 13], "labels": [10, 11, 12, 13]},
        ],
        expected_input_ids=[[10, 11, 0, 0], [10, 11, 12, 13]],
        expected_labels=[[10, 11, -100, -100], [10, 11, 12, 13]],
        expected_attention_mask=[[1, 1, 0, 0], [1, 1, 1, 1]],
    ),
    PaddingAndMaskingTestCase(
        name="internal_pad_tokens",
        batch=[
            {"input_ids": [1, 0, 3, 11, 0], "labels": [1, 10, 3, 11, 2]},
        ],
        expected_input_ids=[[1, 0, 3, 11, 0]],
        expected_labels=[[1, 10, 3, 11, 2]],
        expected_attention_mask=[[1, 0, 1, 1, 0]],
    ),
    PaddingAndMaskingTestCase(
        name="mixed_lengths_and_tokens",
        batch=[
            {"input_ids": [1, 10, 2], "labels": [1, 10, 2]},
            {"input_ids": [10, 11], "labels": [10, 11]},
        ],
        expected_input_ids=[[1, 10, 2], [10, 11, 0]],
        expected_labels=[[1, 10, 2], [10, 11, -100]],
        expected_attention_mask=[[1, 1, 1], [1, 1, 0]],
    ),
]


@pytest.mark.parametrize(
    "case",
    PADDING_AND_MASKING_TEST_CASES,
    ids=lambda c: c.name,
)
def test_padding_and_masking(case: PaddingAndMaskingTestCase) -> None:
    """Tests the core functionality of sequence padding and special token masking."""
    collator = CipherDataCollator()
    result = collator(case.batch)

    # Assert tensor equality
    assert torch.equal(result["input_ids"], torch.tensor(case.expected_input_ids, dtype=torch.long))
    assert torch.equal(result["labels"], torch.tensor(case.expected_labels, dtype=torch.long))
    assert torch.equal(result["attention_mask"], torch.tensor(case.expected_attention_mask, dtype=torch.long))
