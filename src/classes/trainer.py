import os

from classes.config import TRANSFORMER_VERSION
import transformers
from transformers import JambaConfig, JambaForCausalLM, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from classes import Config, CipherDataCollator, PretokenizedCipherDataset, JambaCheckpointManager
from utils.logging import get_logger

logger = get_logger(__name__, level=20)


class JambaTrainingPipeline:
    """Pipeline for initializing, configuring, and training the Jamba model."""

    def __init__(self, config: Config) -> None:
        """Initialize the training pipeline with the given configuration."""
        self.cfg = config
        self._verify_transformers_version()
        self._inject_mamba_kernels()
        self.model = self._initialize_model()
        self.trainer = self._setup_trainer()
        self.checkpoint_manager = JambaCheckpointManager(config)

    def _verify_transformers_version(self) -> None:
        """Verify that the transformers version is compatible with the Mamba kernel patch."""
        if not transformers.__version__.startswith(f"{TRANSFORMER_VERSION}."):
            raise RuntimeError(f"Requires v{TRANSFORMER_VERSION}.x, found {transformers.__version__}")

    def _inject_mamba_kernels(self) -> None:
        """Injects Mamba 2.3.0 kernels into the transformers Jamba implementation."""
        try:
            import mamba_ssm.ops.selective_scan_interface as mamba_1_style  # type: ignore
            import transformers.models.jamba.modeling_jamba as jamba_mod
            from causal_conv1d import causal_conv1d_fn, causal_conv1d_update  # type: ignore

            try:
                from mamba_ssm.ops.triton.selective_state_update import selective_state_update  # type: ignore
            except ImportError:
                from mamba_ssm.ops.triton.ssd_combined import (  # type: ignore
                    mamba_split_conv_open_loop_scan_combined as selective_state_update,
                )

            jamba_mod.mamba_inner_fn = mamba_1_style.mamba_inner_fn
            jamba_mod.selective_scan_fn = mamba_1_style.selective_scan_fn
            jamba_mod.selective_state_update = selective_state_update
            jamba_mod.causal_conv1d_fn = causal_conv1d_fn
            jamba_mod.causal_conv1d_update = causal_conv1d_update
            jamba_mod.is_fast_path_available = True
            logger.info("MAMBA 2.3.0 DETECTED: Successfully bridged kernels for Jamba.")
        except Exception as e:
            logger.error(f"Kernel Bridge failed: {e}")

    def _initialize_model(self) -> JambaForCausalLM:
        """Instantiate the Jamba model using the configuration dataclass."""
        logger.info("Initializing Model...")
        jamba_config = JambaConfig(**self.cfg.jamba_config.__dict__)
        model = JambaForCausalLM(jamba_config)
        logger.info(f"Model initialized. Parameters: {model.num_parameters()}")
        return model.bfloat16()

    def _setup_trainer(self) -> Trainer:
        """Configure the Hugging Face Trainer with datasets and arguments."""
        training_args = TrainingArguments(
            output_dir=str(self.cfg.output_dir),
            per_device_train_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.grad_accum,
            gradient_checkpointing=self.cfg.grad_checkpoint,
            num_train_epochs=self.cfg.epochs,
            learning_rate=self.cfg.learning_rate,
            logging_steps=self.cfg.log_steps,
            eval_strategy="steps",
            eval_steps=self.cfg.eval_steps,
            save_strategy="steps",
            save_steps=self.cfg.save_steps,
            bf16=self.cfg.bf16,
            push_to_hub=False,
            report_to="none",
            dataloader_num_workers=self.cfg.dataloader_num_workers,
            ddp_find_unused_parameters=False,
        )

        train_ds = PretokenizedCipherDataset(self.cfg.training_dir, self.cfg)
        eval_ds = PretokenizedCipherDataset(self.cfg.validation_dir, self.cfg)
        data_collator = CipherDataCollator(self.cfg.pad_token_id)

        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
        )

    def run(self) -> None:
        """Execute the training loop, handling checkpoints and final model saving."""
        output_dir_str = str(self.cfg.output_dir)
        last_checkpoint = get_last_checkpoint(output_dir_str)

        if last_checkpoint is not None:
            logger.info(f"Found checkpoint: {last_checkpoint}. Resuming...")
            self.checkpoint_manager.prepare_for_fast_path(output_dir_str)
        else:
            logger.warning("No checkpoint found. Starting training from scratch.")

        self.trainer.train(resume_from_checkpoint=last_checkpoint)

        if self.trainer.is_world_process_zero():
            logger.info("Training complete. Saving final model to stop auto-chaining.")
            final_save_path = os.path.join(output_dir_str, "final_model")
            self.trainer.save_model(final_save_path)
