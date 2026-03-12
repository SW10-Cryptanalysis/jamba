import os
import torch
from transformers.trainer_utils import get_last_checkpoint
from config import cfg # Added import to fetch correct default dir

def fuse_jamba_experts(state_dict, num_experts=8, moe_layers=(1, 3, 5, 7)):
    """Fuses individual MoE weights into the single blocks required by Jamba's Fast Path."""
    new_state_dict = state_dict.copy()
    fused_count = 0

    for layer in moe_layers:
        if f"model.layers.{layer}.feed_forward.experts.0.gate_proj.weight" not in state_dict:
            continue

        gate_projs, up_projs, down_projs = [], [], []

        for i in range(num_experts):
            gate_key = f"model.layers.{layer}.feed_forward.experts.{i}.gate_proj.weight"
            up_key = f"model.layers.{layer}.feed_forward.experts.{i}.up_proj.weight"
            down_key = f"model.layers.{layer}.feed_forward.experts.{i}.down_proj.weight"

            gate_projs.append(state_dict[gate_key])
            up_projs.append(state_dict[up_key])
            down_projs.append(state_dict[down_key])

            del new_state_dict[gate_key]
            del new_state_dict[up_key]
            del new_state_dict[down_key]

        gate_up_fused = torch.stack([torch.cat([g, u], dim=0) for g, u in zip(gate_projs, up_projs)])
        new_state_dict[f"model.layers.{layer}.feed_forward.experts.gate_up_proj"] = gate_up_fused

        down_fused = torch.stack(down_projs)
        new_state_dict[f"model.layers.{layer}.feed_forward.experts.down_proj"] = down_fused

        fused_count += 1

    return new_state_dict, fused_count

def prepare_checkpoint_for_fast_path(output_dir=str(cfg.output_dir)):
    """Finds the latest checkpoint on disk and fuses it if necessary."""
    latest_ckpt = get_last_checkpoint(output_dir)

    if not latest_ckpt:
        print("No checkpoint found. Starting fresh.")
        return

    safetensors_path = os.path.join(latest_ckpt, "model.safetensors")
    pytorch_path = os.path.join(latest_ckpt, "pytorch_model.bin")

    try:
        from safetensors.torch import load_file, save_file
        has_safetensors = True
    except ImportError:
        has_safetensors = False

    if os.path.exists(safetensors_path) and has_safetensors:
        print(f"Inspecting safetensors checkpoint at: {latest_ckpt}")
        state_dict = load_file(safetensors_path)
        new_state_dict, fused_count = fuse_jamba_experts(state_dict, num_experts=cfg.num_experts)

        if fused_count > 0:
            print(f"Fusing {fused_count} layers into Fast Path format and overwriting checkpoint...")
            save_file(new_state_dict, safetensors_path)
        else:
            print("Checkpoint is already fused or has no unfused MoE layers.")

    elif os.path.exists(pytorch_path):
        print(f"Inspecting PyTorch checkpoint at: {latest_ckpt}")
        state_dict = torch.load(pytorch_path, map_location="cpu")
        new_state_dict, fused_count = fuse_jamba_experts(state_dict, num_experts=cfg.num_experts)

        if fused_count > 0:
            print(f"Fusing {fused_count} layers into Fast Path format and overwriting checkpoint...")
            torch.save(new_state_dict, pytorch_path)
        else:
            print("Checkpoint is already fused or has no unfused MoE layers.")