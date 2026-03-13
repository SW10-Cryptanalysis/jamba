import torch
import os
from datasets import load_from_disk
from torch.utils.data import Dataset
from config import cfg
from utils.logging import get_logger
logger = get_logger(__name__, level=20)

class PretokenizedCipherDataset(Dataset):
    """A PyTorch Dataset class for loading pre-tokenized cipher data from disk.

    Attributes:
        hf_dataset (datasets.Dataset): The Hugging Face Dataset object containing the
            pre-tokenized data.

    """

    def __init__(self, directory_path: str) -> None:
        """Initialize the dataset by loading pre-tokenized data from disk.

        Args:
            directory_path (str): Path to the directory containing the pre-tokenized
                dataset.

        """
        self.hf_dataset = load_from_disk(str(directory_path))
        if len(self.hf_dataset) == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.warning(f"Dataset at {directory_path} is empty.")

    def __len__(self) -> int:
      """Get length of the dataset.

      Returns:
          int: The number of samples in the dataset.

      """
      return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single item from the dataset and applies masking to the labels.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing 'input_ids' and 'labels'
                tensors, where 'labels' has been masked

        """
        item = self.hf_dataset[idx]

        input_ids = item["input_ids"][:cfg.max_context]

        labels = item["labels"][:cfg.max_context] if "labels" in item else list(input_ids)

        # Ensure labels are tensors for masking logic
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)

        # Apply specific masking for Jamba:
        # 1. Mask everything up to (and including) SEP
        sep_indices = (input_ids_t == cfg.sep_token_id).nonzero(as_tuple=True)[0]

        if len(sep_indices) > 0:
            labels_t[:sep_indices[0] + 1] = -100
        else:
            # Log first 10 samples and check local rank to avoid spamming logs
            if idx < 10 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                logger.warning(
                    f"Sample {idx}: No separator token (ID: {cfg.sep_token_id}) found. "
                    f"No prefix masking applied to labels.",
                )

        # 2. Mask special tokens (BOS, EOS, SPACE) so they don't contribute to loss
        # Note: PAD is handled by the collator
        filler_tokens = [cfg.bos_token_id, cfg.eos_token_id, cfg.space_token_id]
        for t_id in filler_tokens:
            labels_t[input_ids_t == t_id] = -100

        return {
            "input_ids": input_ids_t,
            "labels": labels_t,
        }

def safe_pad_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length sequences.

    Ensures that the special tokens are masked out in the labels.

    Args:
      batch (list[dict[str, torch.Tensor]]): A list of samples, where each sample is a
        dict containing 'input_ids' and 'labels' tensors.

    Returns:
      dict[str, torch.Tensor]: A dict containing padded 'input_ids', 'attention_mask',
        and 'labels' tensors ready for model input.

    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=cfg.pad_token_id,
    )

    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100,
    )

    # Jamba needs an attention mask to ignore the PAD tokens
    attention_mask = (input_ids_padded != cfg.pad_token_id).long()

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded,
    }

def prepare_data(directory_path: str) -> Dataset:
    """Match your train.py call signature."""
    return PretokenizedCipherDataset(directory_path)
