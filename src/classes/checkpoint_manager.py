import torch
from pathlib import Path
from typing import Callable, Any

try:
    from safetensors.torch import load_file, save_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from src.classes.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__, level=20)


class JambaCheckpointManager:
    """Manages the inspection, modification, and saving of Jamba model checkpoints.

    Attributes:
        config (Config): The system configuration detailing model architecture.

    """

    def __init__(self, config: Config) -> None:
        """Initialize the checkpoint manager with the system configuration."""
        self.config = config

    def fuse_experts(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
        """Fuse individual MoE weights into single blocks for Fast Path execution.

        Args:
            state_dict (dict[str, torch.Tensor]): The loaded PyTorch state dictionary.

        Returns:
            tuple[dict[str, torch.Tensor], int]: The updated state dictionary and
                the number of fused layers.

        """
        new_state_dict = state_dict.copy()
        fused_count = 0
        layers = self.config.jamba_config.num_hidden_layers
        experts = self.config.jamba_config.num_experts

        for layer in range(layers):
            base_key = f"model.layers.{layer}.feed_forward.experts"

            if f"{base_key}.0.gate_proj.weight" not in state_dict:
                continue

            gate_projs, up_projs, down_projs = [], [], []

            for i in range(experts):
                gate_key = f"{base_key}.{i}.gate_proj.weight"
                up_key = f"{base_key}.{i}.up_proj.weight"
                down_key = f"{base_key}.{i}.down_proj.weight"

                gate_projs.append(state_dict[gate_key])
                up_projs.append(state_dict[up_key])
                down_projs.append(state_dict[down_key])

                del new_state_dict[gate_key]
                del new_state_dict[up_key]
                del new_state_dict[down_key]

            gate_up_fused = torch.stack([torch.cat([g, u], dim=0) for g, u in zip(gate_projs, up_projs, strict=False)])
            new_state_dict[f"{base_key}.gate_up_proj"] = gate_up_fused

            down_fused = torch.stack(down_projs)
            new_state_dict[f"{base_key}.down_proj"] = down_fused

            fused_count += 1

        return new_state_dict, fused_count

    def _process_file(
        self,
        file_path: Path,
        load_fn: Callable[[Path], dict[str, torch.Tensor]],
        save_fn: Callable[[dict[str, torch.Tensor], Path], Any],
        format_name: str,
    ) -> None:
        """Load, fuse, and save a specific checkpoint file format.

        Args:
            file_path (Path): The path to the checkpoint file.
            load_fn (Callable[[Path], dict[str, torch.Tensor]]): The function to load
                the checkpoint file.
            save_fn (Callable[[dict[str, torch.Tensor], Path], Any]): The function to save
                the checkpoint file.
            format_name (str): The name of the checkpoint format.

        """
        logger.info(f"Inspecting {format_name} checkpoint at: {file_path}")
        state_dict = load_fn(file_path)

        new_state_dict, fused_count = self.fuse_experts(state_dict)

        if fused_count > 0:
            logger.debug(f"Fusing {fused_count} layers into Fast Path format and overwriting...")
            save_fn(new_state_dict, file_path)
        else:
            logger.debug("Checkpoint is already fused or has no unfused MoE layers.")

    def prepare_for_fast_path(self, checkpoint_dir: str | Path) -> None:
        """Inspect a checkpoint directory and route it to the correct processor.

        Args:
            checkpoint_dir (str | Path): The directory containing the checkpoint.

        """
        checkpoint_path = Path(checkpoint_dir)

        if not checkpoint_path.exists():
            logger.info("No checkpoint found. Starting fresh.")
            return

        safetensors_path = checkpoint_path / "model.safetensors"
        pytorch_path = checkpoint_path / "pytorch_model.bin"

        if safetensors_path.exists() and HAS_SAFETENSORS:
            self._process_file(
                file_path=safetensors_path,
                load_fn=lambda p: load_file(p),  # type: ignore
                save_fn=lambda d, p: save_file(d, p),  # type: ignore
                format_name="safetensors",
            )
        elif pytorch_path.exists():
            self._process_file(
                file_path=pytorch_path,
                load_fn=lambda p: torch.load(p, map_location="cpu", weights_only=True),
                save_fn=lambda d, p: torch.save(d, p),
                format_name="PyTorch",
            )
