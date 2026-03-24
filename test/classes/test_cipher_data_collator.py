import pytest
import torch
from dataclasses import dataclass

from classes.config import Config
from classes.cipher_data_collator import CipherDataCollator


@dataclass
class CollatorTestCase:
    name: str
    batch_input_ids: list[list[int]]
    batch_labels: list[list[int]]
    expected_input_ids: list[list[int]]
    expected_labels: list[list[int]]
    expected_attention_mask: list[list[int]]


COLLATOR_CASES = [
    CollatorTestCase(
        name="variable_lengths",
        batch_input_ids=[[1, 2, 3], [1, 2]],
        batch_labels=[[4, 5, 6], [4, 5]],
        expected_input_ids=[[1, 2, 3], [1, 2, 0]],  # 0 is our pad_token_id
        expected_labels=[[4, 5, 6], [4, 5, -100]],  # -100 is standard ignore index
        expected_attention_mask=[[1, 1, 1], [1, 1, 0]],
    ),
    CollatorTestCase(
        name="equal_lengths",
        batch_input_ids=[[1, 2], [3, 4]],
        batch_labels=[[5, 6], [7, 8]],
        expected_input_ids=[[1, 2], [3, 4]],
        expected_labels=[[5, 6], [7, 8]],
        expected_attention_mask=[[1, 1], [1, 1]],
    ),
]


@pytest.fixture
def mock_config(mocker):
    """Fixture to provide a Config instance without triggering file IO."""
    mocker.patch.object(Config, "load_homophones")
    return Config(pad_token_id=0)


@pytest.mark.parametrize("case", COLLATOR_CASES, ids=lambda c: c.name)
def test_cipher_data_collator(mock_config, case: CollatorTestCase):
    # Arrange
    collator = CipherDataCollator(config=mock_config)
    batch = [
        {
            "input_ids": torch.tensor(inp),
            "labels": torch.tensor(lbl),
        }
        for inp, lbl in zip(case.batch_input_ids, case.batch_labels, strict=False)
    ]

    # Act
    result = collator(batch)

    # Assert
    assert torch.equal(result["input_ids"], torch.tensor(case.expected_input_ids))
    assert torch.equal(result["labels"], torch.tensor(case.expected_labels))
    assert torch.equal(result["attention_mask"], torch.tensor(case.expected_attention_mask))
