import pytest
from dataclasses import dataclass

from classes.config import Config


@dataclass
class ConfigTestCase:
    name: str
    file_exists: bool
    file_content: str
    expect_error: bool
    expected_homophones: int


CONFIG_TEST_CASES = [
    ConfigTestCase(
        name="valid_metadata",
        file_exists=True,
        file_content='{"max_symbol_id": 1000}',
        expect_error=False,
        expected_homophones=1000,
    ),
    ConfigTestCase(
        name="invalid_json",
        file_exists=True,
        file_content="{malformed,}",
        expect_error=True,
        expected_homophones=500,  # Default fallback before crash
    ),
    ConfigTestCase(
        name="missing_key",
        file_exists=True,
        file_content='{"wrong_key": 100}',
        expect_error=True,
        expected_homophones=500,
    ),
    ConfigTestCase(
        name="file_not_found",
        file_exists=False,
        file_content="",
        expect_error=True,
        expected_homophones=500,
    ),
]


def test_config_properties(mocker):
    # Mock file reading so we can cleanly test the properties without file IO
    mocker.patch("builtins.open", mocker.mock_open(read_data='{"max_symbol_id": 500}'))
    cfg = Config()

    # Verify dynamically calculated properties based on 500 unique homophones
    assert cfg.sep_token_id == 501
    assert cfg.space_token_id == 502
    assert cfg.bos_token_id == 503
    assert cfg.eos_token_id == 504
    assert cfg.char_offset == 505

    # Verify path resolutions
    assert cfg.training_dir == cfg.data_dir / "tokenized_normal" / "Training"
    assert cfg.validation_dir == cfg.data_dir / "tokenized_normal" / "Validation"


@pytest.mark.parametrize("case", CONFIG_TEST_CASES, ids=lambda c: c.name)
def test_config_load_homophones(mocker, case: ConfigTestCase):
    # Setup mocks for standard file operations
    if case.file_exists:
        mocker.patch("builtins.open", mocker.mock_open(read_data=case.file_content))
    else:
        mocker.patch("builtins.open", side_effect=OSError("File not found"))

    # Execute and Assert
    if case.expect_error:
        with pytest.raises(
            RuntimeError,
            match="Aborting initialization: Invalid or missing homophone metadata.",
        ):
            Config()
    else:
        cfg = Config()
        assert cfg.unique_homophones == case.expected_homophones
        assert cfg.jamba_config.vocab_size == cfg.char_offset + cfg.unique_letters + 1
