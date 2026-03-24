import torch

from src.classes.config import Config


class CipherDataCollator:
    """Callable class to pad a batch of variable-length sequences.

    Ensures that the special tokens are masked out in the labels and generates
    the attention mask for the Jamba model to ignore padding.

    Attributes:
        config (Config): Configuration containing the padding token ID.

    """

    def __init__(self, config: Config) -> None:
        """Initialize the collator with the system configuration.

        Args:
            config (Config): System configuration containing token IDs.

        """
        self.config = config

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad a batch of sequences and generate attention masks.

        Args:
            batch (list[dict[str, torch.Tensor]]): A list of dictionary samples.

        Returns:
            dict[str, torch.Tensor]: A dict containing padded 'input_ids',
                'attention_mask', and 'labels' tensors.

        """
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.config.pad_token_id,
        )

        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        attention_mask = (input_ids_padded != self.config.pad_token_id).long()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }
