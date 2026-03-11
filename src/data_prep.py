import torch
import os
from datasets import load_from_disk
from torch.utils.data import Dataset
from config import cfg

class PretokenizedCipherDataset(Dataset):
    def __init__(self, directory_path):
        # We load the existing HF dataset from disk
        self.hf_dataset = load_from_disk(str(directory_path))
        if len(self.hf_dataset) == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Warning: Dataset at {directory_path} is empty.")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        # Pull raw lists from the arrow-backed dataset
        input_ids = item["input_ids"][:cfg.max_context]

        # If your arrow file already has pre-masked labels, use them.
        # Otherwise, we create them here.
        if "labels" in item:
            labels = item["labels"][:cfg.max_context]
        else:
            labels = list(input_ids) # Convert to list for mutation

        # Ensure labels are tensors for masking logic
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)

        # Apply specific masking for Jamba:
        # 1. Mask everything up to (and including) SEP
        sep_indices = (input_ids_t == cfg.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_indices) > 0:
            labels_t[:sep_indices[0] + 1] = -100

        # 2. Mask special tokens (BOS, EOS, SPACE) so they don't contribute to loss
        # Note: PAD is handled by the collator
        filler_tokens = [cfg.bos_token_id, cfg.eos_token_id, cfg.space_token_id]
        for t_id in filler_tokens:
            labels_t[input_ids_t == t_id] = -100

        return {
            "input_ids": input_ids_t,
            "labels": labels_t
        }

def safe_pad_collate(batch):
    """
    Dynamic padding for the batch. This is much faster than static padding.
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=cfg.pad_token_id
    )

    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    # Jamba needs an attention mask to ignore the PAD tokens
    attention_mask = (input_ids_padded != cfg.pad_token_id).long()

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded
    }

def prepare_data(directory_path):
    """Wrapper to match your train.py call signature"""
    return PretokenizedCipherDataset(directory_path)