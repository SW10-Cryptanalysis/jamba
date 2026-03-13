import os
import torch
import transformers
from transformers import JambaConfig, JambaForCausalLM, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint

from config import cfg
from data_prep import prepare_data, safe_pad_collate
from jamba_utils import prepare_checkpoint_for_fast_path
from utils.logging import get_logger
logger = get_logger(__name__, level=20)

# Ensure correct transformers version for Mamba kernel patching
if not transformers.__version__.startswith("5.2."): raise RuntimeError(f"Requires v5.2.x, found {transformers.__version__}")

# Mamba kernel injection
try:
    import mamba_ssm.ops.selective_scan_interface as mamba_1_style
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

    try:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    except ImportError:
        from mamba_ssm.ops.triton.ssd_combined import (
            mamba_split_conv_open_loop_scan_combined as selective_state_update)

    import transformers.models.jamba.modeling_jamba as jamba_mod

    jamba_mod.mamba_inner_fn = mamba_1_style.mamba_inner_fn
    jamba_mod.selective_scan_fn = mamba_1_style.selective_scan_fn
    jamba_mod.selective_state_update = selective_state_update
    jamba_mod.causal_conv1d_fn = causal_conv1d_fn
    jamba_mod.causal_conv1d_update = causal_conv1d_update

    jamba_mod.is_fast_path_available = True
    logger.info("MAMBA 2.3.0 DETECTED: Successfully bridged kernels for Jamba.")
except Exception as e:
    logger.error(f"Kernel Bridge failed: {e}")

# 1. ARCHITECTURE & MODEL

config = JambaConfig(
    vocab_size=cfg.vocab_size,
    hidden_size=cfg.hidden_size,
    num_hidden_layers=cfg.num_hidden_layers,
    num_attention_heads=cfg.num_attention_heads,
    num_key_value_heads=cfg.num_key_value_heads,
    intermediate_size=cfg.intermediate_size,
    attn_layer_period=cfg.attn_layer_period,
    attn_layer_offset=cfg.attn_layer_offset,
    num_experts=cfg.num_experts,
    num_experts_per_tok=cfg.num_experts_per_tok,
    expert_retrieval_size=cfg.expert_retrieval_size,
    max_position_embeddings=cfg.max_context,
    use_mamba_kernels=cfg.use_mamba_kernels,
    use_cache=cfg.use_cache,
)

logger.info("Initializing Model...")
model = JambaForCausalLM(config).to(dtype=torch.bfloat16)

# 2. TRAINING SETUP

training_args = TrainingArguments(
    output_dir=str(cfg.output_dir),
    per_device_train_batch_size=cfg.batch_size,
    gradient_accumulation_steps=cfg.grad_accum,
    gradient_checkpointing=cfg.grad_checkpoint,
    num_train_epochs=cfg.epochs,
    learning_rate=cfg.learning_rate,
    logging_steps=cfg.logging_steps,
    eval_strategy="steps",
    eval_steps=cfg.eval_steps,
    save_strategy="steps",
    save_steps=cfg.save_steps,
    bf16=cfg.bf16,
    push_to_hub=False,
    report_to="none",
    dataloader_num_workers=cfg.dataloader_num_workers,
    ddp_find_unused_parameters=False,
)

train_ds = prepare_data(cfg.training_dir)
eval_ds = prepare_data(cfg.validation_dir)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=safe_pad_collate,
)


# 3. EXECUTION
last_checkpoint = get_last_checkpoint(str(cfg.output_dir))

if last_checkpoint is not None:
    logger.info(f"✅ Found checkpoint: {last_checkpoint}. Resuming...")
    prepare_checkpoint_for_fast_path(str(cfg.output_dir))
else:
    logger.warning("🆕 No checkpoint found. Starting training from scratch.")

trainer.train(resume_from_checkpoint=last_checkpoint)


# 4. FINAL SAVE
if trainer.is_world_process_zero():
    logger.info("🏁 Training complete. Saving final model to stop auto-chaining.")
    final_save_path = os.path.join(str(cfg.output_dir), "final_model")
    trainer.save_model(final_save_path)
