# Jamba Cipher Solver

This project implements and trains a Jamba-like neural network model to solve homophonic substitution ciphers. The model architecture is a hybrid of Mamba (State Space Model) and Transformer blocks, optimized for sequence-to-sequence tasks like decryption.

## Features

- **Hybrid Jamba-Mamba-Transformer Model:** A custom implementation of a Jamba-like architecture.
- **Homophonic Cipher Decryption:** Specifically designed to translate homophonic cipher text into plaintext.
- **Distributed Training:** Supports multi-GPU training using `torchrun`.
- **Configurable:** Easily configure the model architecture, training hyperparameters, and data paths.
- **Training and Evaluation:** Separate scripts for training and evaluating the model.

## Project Structure

```
.
├── src/
│   ├── classes/
│   │   ├── config.py           # Configuration for model and training
│   │   ├── trainer.py          # Training pipeline
│   │   └── evaluator.py        # Evaluation pipeline
│   ├── train.py              # Main script to start training
│   └── eval.py               # Main script to start evaluation
├── test/                     # Pytest tests
├── pyproject.toml            # Project dependencies and settings
├── flake.nix                 # Nix flake for environment setup
└── README.md                 # This file
```

## Installation

This project uses `uv` for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd jamba
    ```

2.  **Install dependencies:**
    ```bash
    uv pip install -e .
    ```
    For development dependencies (like `pytest`):
    ```bash
    uv pip install -e '.[dev]'
    ```

## Configuration

The main configuration is handled by the `Config` and `JambaConfig` data classes in `src/classes/config.py`.

- **`JambaConfig`**: Defines the model architecture (e.g., `hidden_size`, `num_hidden_layers`, `num_experts`).
- **`Config`**: Defines training and evaluation parameters (e.g., `batch_size`, `learning_rate`, data paths).

You can modify this file to change hyperparameters and other settings.

## Usage

The primary method for running this project is through the provided SLURM scripts, which ensure the correct environment and resource allocation on a High-Performance Computing (HPC) cluster.

### Running with SLURM

-   **To start training:** Submit the training job using `sbatch`.
    ```bash
    sbatch run_training.slurm
    ```
-   **To run evaluation:** Submit the evaluation job.
    ```bash
    sbatch run_eval.slurm
    ```

### Execution Details

The SLURM scripts handle the setup of a `Singularity` container and execute the Python scripts within it. You may need to adapt the paths and container details in the `.slurm` files for your specific environment.

-   **Training:** The `run_training.slurm` script uses `torchrun` for distributed, multi-GPU training. The core command is:
    ```bash
    torchrun --nproc_per_node=4 src/train.py
    ```
-   **Evaluation:** The `run_eval.slurm` script executes the evaluation script on the final model:
    ```bash
    python src/eval.py
    ```

While the scripts are designed for a SLURM/Singularity environment, the commands above are the underlying execution logic for training and evaluation. The training script (`run_training.slurm`) also includes a simple auto-chaining logic to re-queue the job if it doesn't complete.

## Dependencies

Key dependencies include:
- `torch`
- `transformers`
- `mamba-ssm`
- `accelerate`
- `datasets`
- `scikit-learn`
- `sentencepiece`

See `pyproject.toml` for a full list of dependencies.
