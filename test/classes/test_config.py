import pytest
from dataclasses import dataclass
from pathlib import Path

from classes.config import Config, MAX_PLAIN_NORMAL, MAX_PLAIN_SPACES, BUFFER


@dataclass
class ConfigTestCase:
    name: str
    file_exists: bool
    file_content: str
    expected_error: type[Exception] | None
    expected_error_match: str | None
    expected_homophones: int


CONFIG_TEST_CASES = [
    ConfigTestCase(
        name="valid_metadata",
        file_exists=True,
        file_content='{"max_symbol_id": 1000}',
        expected_error=None,
        expected_error_match=None,
        expected_homophones=1000,
    ),
    ConfigTestCase(
        name="invalid_json",
        file_exists=True,
        file_content="{malformed,}",
        expected_error=ValueError,
        expected_error_match="Invalid or missing 'max_symbol_id'",
        expected_homophones=500,  # Default fallback before crash
    ),
    ConfigTestCase(
        name="missing_key",
        file_exists=True,
        file_content='{"wrong_key": 100}',
        expected_error=ValueError,
        expected_error_match="Invalid or missing 'max_symbol_id'",
        expected_homophones=500,
    ),
    ConfigTestCase(
        name="file_not_found",
        file_exists=False,
        file_content="",
        expected_error=FileNotFoundError,
        expected_error_match="Cannot determine unique_homophones — aborting",
        expected_homophones=500,
    ),
]

@pytest.mark.parametrize("use_spaces, expected_max, expected_folder", [
    (False, (MAX_PLAIN_NORMAL * 2) + BUFFER, "jamba-cipher-results"),
    (True, (MAX_PLAIN_SPACES * 2) + BUFFER, "jamba-cipher-results-spaced"),
], ids=["normal_mode", "spaced_mode"])
def test_config_dynamic_properties(mocker, use_spaces, expected_max, expected_folder):
    """Test max_context and output_dir based on the use_spaces toggle."""
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("builtins.open", mocker.mock_open(read_data='{"max_symbol_id": 500}'))

    # We set use_spaces via the constructor
    cfg = Config(use_spaces=use_spaces)

    assert cfg.max_context == expected_max
    assert cfg.output_dir.name == expected_folder
    # Ensure it's correctly relative to the project structure
    assert isinstance(cfg.output_dir, Path)

@pytest.mark.parametrize("homophones, vocab_size, expected_valid", [
    (100, 132, True),   # Everything initialized
    (0, 32, False),     # Missing homophones
    (100, 0, False),    # Manual vocab override to 0
], ids=["valid_state", "invalid_homophones", "invalid_vocab"])
def test_is_valid_init(mocker, homophones, vocab_size, expected_valid):
    """Test the safety check that ensures the model is ready for training."""
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("builtins.open", mocker.mock_open(read_data=f'{{"max_symbol_id": {homophones}}}'))

    cfg = Config()
    # Manually override vocab_size to test the boundary case
    cfg.jamba_config.vocab_size = vocab_size

    # max_context is checked here too; it will be > 0 due to BUFFER
    assert cfg.is_valid_init == expected_valid

def test_config_properties(mocker):
    # Mock BOTH path existence and file reading
    mocker.patch("os.path.exists", return_value=True)
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
    # Mock os.path.exists dynamically based on the test case
    mocker.patch("os.path.exists", return_value=case.file_exists)

    # Setup mocks for standard file operations
    if case.file_exists:
        mocker.patch("builtins.open", mocker.mock_open(read_data=case.file_content))
    else:
        mocker.patch("builtins.open", side_effect=OSError("File not found"))

    # Execute and Assert
    if case.expected_error:
        with pytest.raises(case.expected_error, match=case.expected_error_match):
            Config()
    else:
        cfg = Config()
        assert cfg.unique_homophones == case.expected_homophones
        assert cfg.jamba_config.vocab_size == cfg.char_offset + cfg.unique_letters + 1
