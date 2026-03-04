import os
import torch
import mamba_ssm
from mamba_ssm import Mamba
from pathlib import Path
from transformers import JambaConfig, JambaForCausalLM, TrainingArguments, Trainer
from data_prep import prepare_data

# --- DIAGNOSTIC PRINT ---
# This lets us verify the environment without crashing if something is wrong
try:
    import mamba_ssm
    import causal_conv1d
    print(f"✅ Environment Check: Mamba-ssm {mamba_ssm.__version__}, Causal-conv1d {causal_conv1d.__version__}")
except ImportError as e:
    print(f"⚠️ Environment Warning: Kernels not found ({e}). Training will be slow.")

# 1. CONFIGURATION
# We explicitly disable cache to save VRAM for the 10k sequence length
config = JambaConfig(
    vocab_size=4096,
    hidden_size=256,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=2,
    intermediate_size=1024,
    attn_layer_period=4,
    attn_layer_offset=0,
    num_experts=8,
    num_experts_per_tok=2,
    expert_retrieval_size=256,
    max_position_embeddings=10240,

    use_mamba_kernels=True, # Attempt to use kernels
    use_cache=False
)

# 2. MODEL INITIALIZATION
print("Initializing Model...")
model = JambaForCausalLM(config).to("cuda")

# 3. TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./jamba-cipher-results",

    # Batch size tweaks for 10k context
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,

    num_train_epochs=3,
    learning_rate=3e-4,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    bf16=True,
    push_to_hub=False,
    report_to="none",
    dataloader_num_workers=4
)

# 4. DATA PREP
data_dir = Path(__file__).parent.parent.parent / "Ciphers"
train_data_dir = data_dir / "Training"
valid_data_dir = data_dir / "Validation"

print("Initializing Datasets...")
MAX_SEQ_LEN = 10240
train_ds = prepare_data(train_data_dir, max_len=MAX_SEQ_LEN)
eval_ds = prepare_data(valid_data_dir, max_len=MAX_SEQ_LEN)

# 5. TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

print("Starting Training...")
trainer.train()