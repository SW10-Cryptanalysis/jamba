from dataclasses import dataclass
from pathlib import Path

# Context sizing based on your original 10240 limit
TEXT_LEN = 5120  
TOTAL_SEQ = TEXT_LEN * 2
BUFFER = 0 

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "Ciphers"
OUTPUT_DIR = Path(__file__).parent.parent / "jamba-cipher-results"

TRAINING_DIR = DATA_DIR / "Training"
VALIDATION_DIR = DATA_DIR / "Validation"

@dataclass
class Config:
    # ARCHITECTURE
    vocab_size: int = 4096 
    max_context: int = TOTAL_SEQ + BUFFER # 10240
    
    # Token IDs (Mapped to your original dataset structure)
    pad_token_id: int = 3027
    bos_token_id: int = 3028
    eos_token_id: int = 3029
    sep_token_id: int = 3030
    
    # Jamba Specific Hyperparameters
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
    use_mamba_kernels: bool = True
    use_cache: bool = False
    
    # TRAINING
    batch_size: int = 2 
    grad_accum: int = 8
    learning_rate: float = 3e-4
    epochs: int = 3
    
    grad_checkpoint: bool = True
    bf16: bool = True
    dataloader_num_workers: int = 4
    
    # STEPS
    logging_steps: int = 50
    save_steps: int = 1000
    eval_steps: int = 1000

    # SYSTEM
    output_dir: Path = OUTPUT_DIR
    training_dir: Path = TRAINING_DIR
    validation_dir: Path = VALIDATION_DIR

cfg = Config()