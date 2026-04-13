import torch
from typing import Any
from torch.nn.utils.rnn import pad_sequence
from classes.config import Config


class CipherDataCollator:
    """Callable class to pad a batch of variable-length sequences.

    Truncates sequences to max_context, applies masking to special filler tokens,
    pads the sequences, and generates the attention mask for the Jamba model.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the collator with the given configuration."""
        self.config = config
        self.ignore_index = -100

    def _truncate(self, seq: list[int]) -> list[int]:
        """Truncate sequences based on config max context."""
        if self.config.max_context is None:
            return seq
        return seq[: self.config.max_context]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of samples by truncating, masking, padding, and generating attention masks."""
        if not batch:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "labels": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
            }

        filler_tokens = [
            self.config.bos_token_id,
            self.config.eos_token_id,
            self.config.space_token_id,
        ]

        input_tensors = []
        label_tensors = []

        for item in batch:
            # 1. Truncate
            inp = self._truncate(item["input_ids"])
            lab = self._truncate(item["labels"])

            # 2. Convert to tensors
            inp_t = torch.tensor(inp, dtype=torch.long)
            lab_t = torch.tensor(lab, dtype=torch.long)

            # 3. Mask special filler tokens
            for t_id in filler_tokens:
                lab_t[inp_t == t_id] = self.ignore_index

            input_tensors.append(inp_t)
            label_tensors.append(lab_t)

        # 4. Pad sequences
        input_ids_padded = pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=self.config.pad_token_id,
        )

        labels_padded = pad_sequence(
            label_tensors,
            batch_first=True,
            padding_value=self.ignore_index,
        )

        # 5. Generate attention mask
        attention_mask = (input_ids_padded != self.config.pad_token_id).long()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }
