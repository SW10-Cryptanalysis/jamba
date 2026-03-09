import os
import torch
import mamba_ssm
from mamba_ssm import Mamba
from pathlib import Path
from transformers import JambaConfig, JambaForCausalLM, TrainingArguments, Trainer
from data_prep import prepare_data
from jamba_utils import prepare_checkpoint_for_fast_path

# Checks for mamba kernels
try:
    # Mamba 2.x moved these around. We import them from the new locations
    # and map them to what Jamba (transformers) expects.
    import mamba_ssm.ops.selective_scan_interface as mamba_1_style
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

    try:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    except ImportError:
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv_open_loop_scan_combined as selective_state_update

    import transformers.models.jamba.modeling_jamba as jamba_mod

    # Injecting the kernels from the v2.3.0 installation
    jamba_mod.mamba_inner_fn = mamba_1_style.mamba_inner_fn
    jamba_mod.selective_scan_fn = mamba_1_style.selective_scan_fn
    jamba_mod.selective_state_update = selective_state_update
    jamba_mod.causal_conv1d_fn = causal_conv1d_fn
    jamba_mod.causal_conv1d_update = causal_conv1d_update

    jamba_mod.is_fast_path_available = True
    print("🚀 MAMBA 2.3.0 DETECTED: Successfully bridged kernels for Jamba.")
except Exception as e:
    print(f"⚠️ Kernel Bridge failed: {e}")

# 1. CONFIGURATION
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

    use_mamba_kernels=True,
    use_cache=False
)

# 2. MODEL INITIALIZATION
print("Initializing Model...")
model = JambaForCausalLM(config).to(dtype=torch.bfloat16, device="cuda")

# 3. TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./jamba-cipher-results",

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

# 6. FIND CHECKPOINT
print("Preparing checkpoint for Fast Path compatibility...")
prepare_checkpoint_for_fast_path(training_args.output_dir)

print("Starting Training...")
trainer.train(resume_from_checkpoint=True)