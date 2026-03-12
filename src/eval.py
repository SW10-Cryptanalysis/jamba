import torch
from torch.utils.data import DataLoader
from transformers import JambaForCausalLM
from tqdm import tqdm

from config import cfg
from data_prep import prepare_data, safe_pad_collate
from utils.logging import get_logger

logger = get_logger(__name__, level=20)


def run_evaluation() -> None:
	"""Run evaluation on the validation set and computes Symbol Error Rate (SER)."""
	model_path = cfg.output_dir / "final_model"
	if not model_path.exists():
		logger.error(f"Could not find model at {model_path}. Is training finished?")
		return

	logger.info(f"Loading final model from {model_path}...")
	model = JambaForCausalLM.from_pretrained(
		model_path,
		torch_dtype=torch.bfloat16,
		device_map="auto",
	)
	model.eval()

	logger.debug(f"Loading validation data from {cfg.validation_dir}...")
	val_ds = prepare_data(cfg.test_dir)
	val_loader = DataLoader(
		val_ds,
		batch_size=cfg.batch_size,
		collate_fn=safe_pad_collate,
		num_workers=4,
	)

	total_errors = 0
	total_symbols = 0
	sample_outputs = []

	logger.info("Starting validation evaluation...")
	with torch.no_grad():
		for i, batch in enumerate(tqdm(val_loader)):
			input_ids = batch["input_ids"].to("cuda")
			labels = batch["labels"].to("cuda")

			outputs = model(input_ids=input_ids)
			preds = torch.argmax(outputs.logits, dim=-1)

			mask = labels != -100

			if mask.any():
				total_errors += (preds[mask] != labels[mask]).sum().item()
				total_symbols += mask.sum().item()

			if i == 0:
				for j in range(min(len(input_ids), 3)):
					sample_outputs.append(
						{
							"truth": labels[j][mask[j]].tolist(),
							"pred": preds[j][mask[j]].tolist(),
						},
					)

	final_ser = total_errors / total_symbols if total_symbols > 0 else 0

	logger.info("\n" + "=" * 50)
	logger.info(f"FINAL VALIDATION SER: {final_ser:.6f}")
	logger.info("=" * 50 + "\n")

	logger.info("📋 Qualitative Sample (Token IDs):")
	for idx, sample in enumerate(sample_outputs):
		logger.info(f"\nExample {idx + 1}:")
		logger.info(f"Target: {sample['truth'][:20]}...")
		logger.info(f"Model:  {sample['pred'][:20]}...")


if __name__ == "__main__":
	run_evaluation()
