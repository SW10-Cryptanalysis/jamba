import torch
from pathlib import Path
import os
from datasets import load_from_disk
from torch.utils.data import Dataset

from classes.config import Config
from utils.logging import get_logger

logger = get_logger(__name__, level=20)


class PretokenizedCipherDataset(Dataset):
    """A PyTorch Dataset class for loading pre-tokenized cipher data from disk.

    Attributes:
        hf_dataset (datasets.Dataset): The Hugging Face Dataset object containing the
            pre-tokenized data.
        config (Config): The configuration object containing token IDs and limits.

    """

    def __init__(self, directory_path: str | Path, config: Config) -> None:
        """Initialize the dataset by loading pre-tokenized data from disk.

        Args:
            directory_path (str | Path): Path to the directory containing the dataset.
            config (Config): System configuration containing token IDs and model bounds.

        """
        self.config = config
        self.hf_dataset = load_from_disk(str(directory_path))

        if len(self.hf_dataset) == 0 and self._is_main_process():
            logger.warning(f"Dataset at {directory_path} is empty.")

    def _is_main_process(self) -> bool:
        """Check if the current process is the main process.

        Returns:
            bool: True if the current process is the main process, False otherwise.

        """
        rank = os.environ.get("LOCAL_RANK", "0")
        return rank.isdigit() and int(rank) == 0

    def __len__(self) -> int:
        """Get length of the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single item from the dataset and applies masking to the special tokens.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing 'input_ids' and 'labels'
                tensors, where 'labels' has been masked.

        """
        item = self.hf_dataset[idx]
        labels = item["labels"] if "labels" in item else list(item["input_ids"])

        return {
            "input_ids": item["input_ids"],
            "labels": labels,
        }
