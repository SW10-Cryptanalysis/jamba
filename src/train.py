import sys
import os

# 1. SURGICAL PATH CLEANUP
# This removes the container's global site-packages so Python CANNOT see the broken torchao
sys.path = [p for p in sys.path if "dist-packages" not in p]

# 2. GHOSTING
# Now we tell Python these modules definitely don't exist
sys.modules["torchvision"] = None
sys.modules["torchvision.ops"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchao"] = None  # This stops the 'AttributeError' you just saw

# 3. PROCEED WITH IMPORTS
import torch
from pathlib import Path
from transformers import JambaConfig, JambaForCausalLM, TrainingArguments, Trainer
from data_prep import prepare_data

print("Shields active. System-wide torchao and torchvision blocked.")

# 2. CONFIG: The "True" Jamba Architecture (Updated for Head Consistency)
config = JambaConfig(
    vocab_size=4096,
    hidden_size=256,
    num_hidden_layers=8,

    # --- FIXED ATTENTION HEADS ---
    num_attention_heads=8,        # Increased to 8
    num_key_value_heads=2,        # Explicitly set (8 is divisible by 2)
    # Head dimension will be 256 / 8 = 32 (Perfectly fine)

    intermediate_size=1024,
    attn_layer_period=4,
    attn_layer_offset=0,
    num_experts=8,
    num_experts_per_tok=2,
    expert_retrieval_size=256,

    max_position_embeddings=10240,
    use_mamba_kernels=True
)

model = JambaForCausalLM(config)

# 3. TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./jamba-cipher-results",

    # Memory Management for 10k sequences on 24GB VRAM
    per_device_train_batch_size=2,  # Start small (2 or 4) for 10k lengths
    gradient_accumulation_steps=8,  # Effective batch size = 16
    gradient_checkpointing=True,    # Saves VRAM at the cost of slight compute time

    num_train_epochs=3,             # Reduced from 10; 1M files x 3 epochs is plenty
    learning_rate=3e-4,             # Slightly lower LR is safer for MoE training

    logging_steps=50,
    eval_strategy="steps",          # Evaluate periodically, not just at the end of 1M files
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,

    bf16=True,                      # bfloat16 is heavily recommended for MoE stability (if Ampere+ GPU)
    # fp16=True,                    # Use this instead if you have an older GPU (T4, V100)

    push_to_hub=False,
    report_to="none",
    dataloader_num_workers=4        # Speed up data loading for 1M files
)

# 4. LOAD DATA
data_dir = Path(__file__).parent.parent.parent / "Ciphers"
train_data_dir = data_dir / "Training"
valid_data_dir = data_dir / "Validation"

print("Initializing Datasets...")
MAX_SEQ_LEN = 10240
train_ds = prepare_data(train_data_dir, max_len=MAX_SEQ_LEN)
eval_ds = prepare_data(valid_data_dir, max_len=MAX_SEQ_LEN)

# 5. INITIALIZE & TRAIN
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

print("Starting Training...")
trainer.train()