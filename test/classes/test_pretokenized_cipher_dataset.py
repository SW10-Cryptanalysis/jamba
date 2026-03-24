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


@pytest.fixture
def mock_config(mocker):
    """Fixture providing a mock Config with predictable token IDs."""
    mocker.patch.object(Config, "load_homophones")
    cfg = Config()
    """
    Since unique_homophones defaults to 500, the properties naturally compute to:
    sep_token_id = 501
    space_token_id = 502
    bos_token_id = 503
    eos_token_id = 504
    """
    return cfg


@pytest.mark.parametrize("case", INIT_CASES, ids=lambda c: c.name)
def test_dataset_init_and_len(mocker, mock_config, case: InitTestCase):
    """Test dataset initialization, rank checking, and length retrieval."""
    mock_hf_dataset = MagicMock()
    mock_hf_dataset.__len__.return_value = case.dataset_len
    mocker.patch("classes.pretokenized_cipher_dataset.load_from_disk", return_value=mock_hf_dataset)

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
    """Data for testing the getitem sequence truncation and masking logic."""

    name: str
    item_dict: dict[str, list[int]]
    max_len: int
    idx: int
    local_rank: str
    expected_labels: list[int]
    expect_warning: bool


GETITEM_CASES = [
    GetItemTestCase(
        name="standard_masking_with_sep_and_fillers",
        item_dict={"input_ids": [1, 2, 501, 3, 503, 4], "labels": [10, 20, 30, 40, 50, 60]},
        max_len=100,
        idx=0,
        local_rank="0",
        expected_labels=[-100, -100, -100, 40, -100, 60],
        expect_warning=False,
    ),
    GetItemTestCase(
        name="no_labels_in_item_uses_input_ids",
        item_dict={"input_ids": [1, 501, 3]},
        max_len=100,
        idx=0,
        local_rank="0",
        expected_labels=[-100, -100, 3],
        expect_warning=False,
    ),
    GetItemTestCase(
        name="truncation_applied_to_max_len",
        item_dict={"input_ids": [1, 501, 3, 4, 5], "labels": [10, 20, 30, 40, 50]},
        max_len=3,
        idx=0,
        local_rank="0",
        expected_labels=[-100, -100, 30],
        expect_warning=False,
    ),
    GetItemTestCase(
        name="no_sep_warns_main_process_early_idx",
        item_dict={"input_ids": [1, 2, 3], "labels": [10, 20, 30]},
        max_len=100,
        idx=5,
        local_rank="0",
        expected_labels=[10, 20, 30],
        expect_warning=True,
    ),
    GetItemTestCase(
        name="no_sep_no_warn_late_idx",
        item_dict={"input_ids": [1, 2, 3], "labels": [10, 20, 30]},
        max_len=100,
        idx=15,
        local_rank="0",
        expected_labels=[10, 20, 30],
        expect_warning=False,
    ),
    GetItemTestCase(
        name="no_sep_no_warn_not_main_process",
        item_dict={"input_ids": [1, 2, 3], "labels": [10, 20, 30]},
        max_len=100,
        idx=5,
        local_rank="1",
        expected_labels=[10, 20, 30],
        expect_warning=False,
    ),
]


@pytest.mark.parametrize("case", GETITEM_CASES, ids=lambda c: c.name)
def test_dataset_getitem(mocker, mock_config, case: GetItemTestCase):
    """Test data retrieval, sequence truncation, and prefix/filler masking rules."""
    mock_hf_dataset = MagicMock()
    mock_hf_dataset.__getitem__.return_value = case.item_dict
    mock_hf_dataset.__len__.return_value = 1
    mocker.patch("classes.pretokenized_cipher_dataset.load_from_disk", return_value=mock_hf_dataset)

    mock_config.jamba_config.max_position_embeddings = case.max_len

    mocker.patch.dict(os.environ, {"LOCAL_RANK": case.local_rank}, clear=True)
    mock_logger = mocker.patch("classes.pretokenized_cipher_dataset.logger.warning")

    dataset = PretokenizedCipherDataset("dummy_path", mock_config)
    result = dataset[case.idx]

    assert torch.equal(result["labels"], torch.tensor(case.expected_labels, dtype=torch.long))

    if case.expect_warning:
        mock_logger.assert_called_once()
    else:
        mock_logger.assert_not_called()
