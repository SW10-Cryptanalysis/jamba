from dataclasses import dataclass, field
from pathlib import Path
import json

from utils.logging import get_logger

logger = get_logger(__name__, level=20)

TRANSFORMER_VERSION = 5.3
TEXT_LEN = 10_000
UNIQUE_HOMOPHONES = 500
TOTAL_SEQ = TEXT_LEN * 2
BUFFER = 8

BASE_DIR = Path(__file__).parent.parent.parent.parent


@dataclass
class JambaConfig:
    """Configuration class specifically for the Jamba model.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        hidden_size (int): Size of the hidden layer.
        num_hidden_layers (int): Number of hidden layers.
        num_attention_heads (int): Number of attention heads.
        num_key_value_heads (int): Number of key-value heads.
        intermediate_size (int): Size of the intermediate layer.
        attn_layer_period (int): Periodicity of attention layers.
        attn_layer_offset (int): Offset for attention layer placement.
        num_experts (int): Total number of experts in MoE layers.
        num_experts_per_tok (int): Number of experts to retrieve per token.
        expert_retrieval_size (int): Size of the retrieval vector for experts.
        max_position_embeddings (int): Maximum context length.
        use_mamba_kernels (bool): Whether to use Mamba kernels.
        use_cache (bool): Whether to use caching in the model.

    """

    vocab_size: int = UNIQUE_HOMOPHONES + 26 + BUFFER
    hidden_size: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    intermediate_size: int = 1024
    attn_layer_period: int = 4
    attn_layer_offset: int = 0
    num_experts: int = 8
    num_experts_per_tok: int = 2
    expert_retrieval_size: int = 256
    max_position_embeddings: int = TOTAL_SEQ + 1
    use_mamba_kernels: bool = True
    use_cache: bool = False


@dataclass
class Config:
    """Configuration class for Jamba Cipher Training and Evaluation.

    Attributes:
        unique_homophones (int): Number of unique homophone symbols.
        unique_letters (int): Number of standard letters in the alphabet.
        pad_token_id (int): ID of the padding token.
        sep_token_id (int): ID of the separator token.
        space_token_id (int): ID of the space token.
        bos_token_id (int): ID of the beginning-of-sequence token.
        eos_token_id (int): ID of the end-of-sequence token.
        char_offset (int): Offset for character tokens.
        jamba_config (JambaConfig): Sub-configuration for the model architecture.
        batch_size (int): Batch size for training and evaluation.
        grad_accum (int): Number of gradient accumulation steps.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        grad_checkpoint (bool): Whether to use gradient checkpointing.
        bf16 (bool): Whether to use bfloat16 precision.
        dataloader_num_workers (int): Number of workers for data loading.
        log_steps (int): Steps interval for logging training progress.
        save_steps (int): Steps interval for saving model checkpoints.
        eval_steps (int): Steps interval for running evaluation.
        output_dir (Path): Directory to save model checkpoints and logs.
        data_dir (Path): Root directory containing cipher datasets.
        training_dir (Path): Directory containing the training dataset.
        validation_dir (Path): Directory containing the validation dataset.
        use_spaces (bool): Whether to train on the spaced dataset.

    """

    unique_homophones: int = 500
    unique_letters: int = 26

    # Token IDs
    pad_token_id: int = 0

    @property
    def sep_token_id(self) -> int:
        """Seperator token."""
        return self.unique_homophones + 1

    @property
    def space_token_id(self) -> int:
        """Space token."""
        return self.sep_token_id + 1

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token."""
        return self.space_token_id + 1

    @property
    def eos_token_id(self) -> int:
        """End of sequence token."""
        return self.bos_token_id + 1

    @property
    def char_offset(self) -> int:
        """Character ofset to avoid clashes with defined tokens."""
        return self.eos_token_id + 1

    jamba_config: JambaConfig = field(default_factory=JambaConfig)

    batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 3e-4
    epochs: int = 3

    grad_checkpoint: bool = True
    bf16: bool = True
    dataloader_num_workers: int = 4

    log_steps: int = 50
    save_steps: int = 1000
    eval_steps: int = 1000
    use_spaces: bool = False

    data_dir: Path = BASE_DIR / "Ciphers"

    @property
    def output_dir(self) -> Path:
        """Directory to save model checkpoints and logs."""
        folder_name = "jamba-cipher-results-spaced" if self.use_spaces else "jamba-cipher-results"
        return Path(__file__).parent.parent.parent / folder_name

    @property
    def training_dir(self) -> Path:
        """Directory containing the training dataset."""
        folder = "tokenized_spaced" if self.use_spaces else "tokenized_normal"
        return self.data_dir / folder / "Training"

    @property
    def validation_dir(self) -> Path:
        """Directory containing the validation dataset."""
        folder = "tokenized_spaced" if self.use_spaces else "tokenized_normal"
        return self.data_dir / folder / "Validation"

    def load_homophones(self, homophone_file: str = "metadata.json") -> None:
        """Load the homophone metadata file and set the unique homophone count."""
        homophone_path = self.data_dir / homophone_file
        try:
            with open(homophone_path) as f:
                meta = json.load(f)
                self.unique_homophones = int(meta["max_symbol_id"])
        except (OSError, ValueError, KeyError) as e:
            logger.error(f"Critical failure loading {homophone_path}. Metadata is required for vocab sizing.")
            raise RuntimeError("Aborting initialization: Invalid or missing homophone metadata.") from e

        self.jamba_config.vocab_size = self.char_offset + self.unique_letters + 1

    def __post_init__(self) -> None:
        """Post init hook."""
        self.load_homophones()
