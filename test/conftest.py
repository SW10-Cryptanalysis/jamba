import sys
from unittest.mock import MagicMock

# Intercept heavy ML library imports during Pytest's collection phase
# This drops test collection time from ~8 seconds to milliseconds.
mock_transformers = MagicMock()
# Ensure the mock naturally passes standard version checks
mock_transformers.__version__ = "5.2.0"

sys.modules["transformers"] = mock_transformers
sys.modules["transformers.trainer_utils"] = MagicMock()
sys.modules["datasets"] = MagicMock()
