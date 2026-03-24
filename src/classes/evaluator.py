import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import JambaForCausalLM
from tqdm import tqdm

from classes import Config, CipherDataCollator, PretokenizedCipherDataset
from utils.logging import get_logger

logger = get_logger(__name__, level=20)


class JambaEvaluator:
    """Pipeline for evaluating the Jamba model on the validation dataset."""

    def __init__(self, config: Config, model_path: Path | None = None) -> None:
        """Initialize the evaluator with configuration and optional model path."""
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or (self.cfg.output_dir / "final_model")

        self.model = self._load_model()
        self.val_loader = self._prepare_dataloader()

    def _load_model(self) -> JambaForCausalLM:
        """Load the trained model from disk and sets it to evaluation mode."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Could not find model at {self.model_path}. Is training finished?")

        logger.info(f"Loading model from {self.model_path}...")
        model = JambaForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        return model

    def _prepare_dataloader(self) -> DataLoader:
        """Instantiate the dataset and dataloader for evaluation."""
        logger.debug(f"Loading validation data from {self.cfg.validation_dir}...")
        val_ds = PretokenizedCipherDataset(self.cfg.validation_dir, self.cfg)
        collator = CipherDataCollator(self.cfg)

        return DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            collate_fn=collator,
            num_workers=self.cfg.dataloader_num_workers,
        )

    def _process_evaluation_batch(
        self,
        batch: dict[str, torch.Tensor],
        extract_samples: bool = False,
    ) -> tuple[int, int, list[dict[str, list[int]]]]:
        """Run the forward pass and compute errors for a single batch.

        Args:
            batch (dict[str, torch.Tensor]): The input batch containing input_ids and labels.
            extract_samples (bool): Whether to extract qualitative sample predictions.

        Returns:
            tuple[int, int, list[dict[str, list[int]]]]: A tuple containing the number of
                errors, the total number of valid symbols, and a list of qualitative samples.

        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self.model(input_ids=input_ids)
        preds = torch.argmax(outputs.logits, dim=-1)

        mask = labels != -100
        errors = 0
        symbols = 0
        samples = []

        if mask.any():
            errors = int((preds[mask] != labels[mask]).sum().item())
            symbols = int(mask.sum().item())

        if extract_samples:
            for j in range(min(len(input_ids), 3)):
                samples.append(
                    {
                        "truth": labels[j][mask[j]].tolist(),
                        "pred": preds[j][mask[j]].tolist(),
                    },
                )

        return errors, symbols, samples

    def evaluate(self) -> float:
        """Run evaluation loop to compute Symbol Error Rate (SER)."""
        total_errors = 0
        total_symbols = 0
        sample_outputs = []

        logger.info("Starting validation evaluation...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader)):
                errors, symbols, samples = self._process_evaluation_batch(
                    batch=batch,
                    extract_samples=(i == 0),
                )

                total_errors += errors
                total_symbols += symbols

                if samples:
                    sample_outputs.extend(samples)

        final_ser = total_errors / total_symbols if total_symbols > 0 else 0.0

        logger.info("\n" + "=" * 50)
        logger.info(f"FINAL VALIDATION SER: {final_ser:.6f}")
        logger.info("=" * 50 + "\n")

        logger.info("Qualitative Sample (Token IDs):")
        for idx, sample in enumerate(sample_outputs):
            logger.info(f"\nExample {idx + 1}:")
            logger.info(f"Target: {sample['truth'][:20]}...")
            logger.info(f"Model:  {sample['pred'][:20]}...")

        return final_ser
