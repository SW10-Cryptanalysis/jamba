import pytest
import torch
import os
from dataclasses import dataclass
from unittest.mock import MagicMock

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

@dataclass
class GetItemTestCase:
    """Data for testing that the dataset correctly fetches raw lists."""
    name: str
    item_dict: dict[str, list[int]]
    expected_input_ids: list[int]
    expected_labels: list[int]

GETITEM_CASES = [
    GetItemTestCase(
        name="fetches_raw_data_correctly",
        item_dict={"input_ids": [1, 2, 3], "labels": [10, 20, 30]},
        expected_input_ids=[1, 2, 3],
        expected_labels=[10, 20, 30],
    ),
    GetItemTestCase(
        name="no_labels_in_item_falls_back_to_input_ids",
        item_dict={"input_ids": [1, 502, 3]},
        expected_input_ids=[1, 502, 3],
        expected_labels=[1, 502, 3],
    ),
]

@pytest.fixture
def mock_config(mocker):
    """Fixture providing a mock Config."""
    # 1. Patch load_homophones on the CLASS before instantiation 
    # to prevent __post_init__ from hitting the disk.
    mocker.patch.object(Config, "load_homophones", return_value=None)
    
    cfg = Config()
    
    # 2. Set fields that have setters (standard dataclass fields)
    cfg.unique_homophones = 500
    cfg.use_spaces = False 
    
    # NOTE: We removed cfg.max_context = 100 because it is a read-only property.
    # Since the Dataset no longer uses it, we don't need to mock it here.
    
    return cfg

@pytest.mark.parametrize("case", INIT_CASES, ids=lambda c: c.name)
def test_dataset_init_and_len(mocker, mock_config, case: InitTestCase):
    """Test dataset initialization, rank checking, and length retrieval."""
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

@pytest.mark.parametrize("case", GETITEM_CASES, ids=lambda c: c.name)
def test_dataset_getitem(mocker, mock_config, case: GetItemTestCase):
    """Test that the dataset retrieves raw lists without modification."""
    mock_hf_dataset = mocker.MagicMock()
    mock_hf_dataset.__getitem__.return_value = case.item_dict
    mocker.patch("classes.pretokenized_cipher_dataset.load_from_disk", return_value=mock_hf_dataset)

    dataset = PretokenizedCipherDataset("dummy_path", mock_config)
    result = dataset[0]

    assert result["input_ids"] == case.expected_input_ids
    assert result["labels"] == case.expected_labels
    assert isinstance(result["input_ids"], list)