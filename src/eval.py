import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import JambaForCausalLM
from tqdm import tqdm

from config import cfg
from data_prep import prepare_data, safe_pad_collate

def run_evaluation():
    # 1. LOAD MODEL
    model_path = cfg.output_dir / "final_model"
    if not model_path.exists():
        print(f"❌ Error: Could not find model at {model_path}. Is training finished?")
        return

    print(f"Loading final model from {model_path}...")
    model = JambaForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    # 2. LOAD VALIDATION DATA
    print(f"Loading validation data from {cfg.validation_dir}...")
    val_ds = prepare_data(cfg.test_dir)
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.batch_size, 
        collate_fn=safe_pad_collate,
        num_workers=4
    )

    # 3. EVALUATION LOOP
    total_errors = 0
    total_symbols = 0
    sample_outputs = []

    print("🚀 Starting validation evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")

            # Forward pass
            outputs = model(input_ids=input_ids)
            # Use argmax to get the most likely tokens
            preds = torch.argmax(outputs.logits, dim=-1)

            # Mask out the -100 (Ciphertext + Padding) to only grade the Plaintext
            mask = labels != -100
            
            # Quantitative Check
            if mask.any():
                total_errors += (preds[mask] != labels[mask]).sum().item()
                total_symbols += mask.sum().item()

            # Qualitative Check: Save a few examples from the first batch
            if i == 0:
                for j in range(min(len(input_ids), 3)):
                    sample_outputs.append({
                        "truth": labels[j][mask[j]].tolist(),
                        "pred": preds[j][mask[j]].tolist()
                    })

    # 4. CALCULATE FINAL SER
    # Formula: $$ SER = \frac{\sum Errors}{\sum Total Symbols} $$
    final_ser = total_errors / total_symbols if total_symbols > 0 else 0
    
    print("\n" + "="*50)
    print(f"FINAL VALIDATION SER: {final_ser:.6f}")
    print("="*50 + "\n")

    # 5. PRINT EXAMPLES
    print("📋 Qualitative Sample (Token IDs):")
    for idx, sample in enumerate(sample_outputs):
        print(f"\nExample {idx + 1}:")
        print(f"Target: {sample['truth'][:20]}...")
        print(f"Model:  {sample['pred'][:20]}...")

if __name__ == "__main__":
    run_evaluation()