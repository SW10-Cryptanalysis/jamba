import pytest
import torch
import os
from dataclasses import dataclass

from classes.config import Config
from classes.pretokenized_cipher_dataset import PretokenizedCipherDataset


@dataclass
class InitTestCase:
    """Data for testing initialization and main process checking."""

    name: str
    dataset_len: int
    local_rank: str | None
    expect_warning: bool


INIT_CASES = [
    InitTestCase("empty_dataset_main_process", 0, "0", True),
    InitTestCase("empty_dataset_not_main", 0, "1", False),
    InitTestCase("empty_dataset_invalid_rank", 0, "invalid", False),
    InitTestCase("empty_dataset_no_rank", 0, None, True),
    InitTestCase("non_empty_dataset", 5, "0", False),
]


@pytest.fixture
def mock_config(mocker):
    """Fixture providing a mock Config with predictable token IDs."""
    mocker.patch.object(Config, "load_homophones")
    return Config()


@pytest.mark.parametrize("case", INIT_CASES, ids=lambda c: c.name)
def test_dataset_init_and_len(mocker, mock_config, case: InitTestCase):
    """Test dataset initialization, rank checking, and length retrieval."""
    # mocker.patch returns a MagicMock by default, which supports __len__
    mock_load = mocker.patch("classes.pretokenized_cipher_dataset.load_from_disk")
    mock_hf_dataset = mock_load.return_value
    mock_hf_dataset.__len__.return_value = case.dataset_len

    mocker.patch.dict(os.environ, {}, clear=True)
    if case.local_rank is not None:
        os.environ["LOCAL_RANK"] = case.local_rank

    mock_logger = mocker.patch("classes.pretokenized_cipher_dataset.logger.warning")

    dataset = PretokenizedCipherDataset("dummy_path", mock_config)

    assert len(dataset) == case.dataset_len

    if case.expect_warning:
        mock_logger.assert_called_once()
    else:
        mock_logger.assert_not_called()


@dataclass
class GetItemTestCase:
    """Data for testing the getitem sequence truncation and filler masking logic."""

    name: str
    item_dict: dict[str, list[int]]
    max_len: int
    expected_labels: list[int]


GETITEM_CASES = [
    GetItemTestCase(
        name="masks_only_filler_tokens",
        item_dict={"input_ids": [1, 2, 501, 3, 503, 4], "labels": [10, 20, 30, 40, 50, 60]},
        max_len=100,
        expected_labels=[10, 20, 30, 40, -100, 60],
    ),
    GetItemTestCase(
        name="no_labels_in_item_uses_input_ids_with_masking",
        item_dict={"input_ids": [1, 502, 3, 504]},
        max_len=100,
        expected_labels=[1, -100, 3, -100],
    ),
    GetItemTestCase(
        name="truncation_applied_before_masking",
        item_dict={"input_ids": [1, 501, 503, 4, 5], "labels": [10, 20, 30, 40, 50]},
        max_len=3,
        expected_labels=[10, 20, -100],
    ),
]


@pytest.mark.parametrize("case", GETITEM_CASES, ids=lambda c: c.name)
def test_dataset_getitem(mocker, mock_config, case: GetItemTestCase):
    """Test data retrieval, truncation, and specifically filler masking."""
    # Using MagicMock for __getitem__ support
    mock_hf_dataset = mocker.MagicMock()
    mock_hf_dataset.__getitem__.return_value = case.item_dict
    mocker.patch("classes.pretokenized_cipher_dataset.load_from_disk", return_value=mock_hf_dataset)

    mock_config.jamba_config.max_position_embeddings = case.max_len
    mocker.patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=True)

    dataset = PretokenizedCipherDataset("dummy_path", mock_config)
    result = dataset[0]

    assert torch.equal(result["labels"], torch.tensor(case.expected_labels, dtype=torch.long))
