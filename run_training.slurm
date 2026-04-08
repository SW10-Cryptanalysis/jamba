#!/bin/bash
#SBATCH --job-name=jamba_cipher
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:l4:4

set -x
#set -e
#set -o pipefail

# --- 1. DEFINE PATHS ---
PROJECT_DIR="/ceph/project/SW10-CausalLM/jamba"
CONTAINER="${PROJECT_DIR}/pytorch_25.03-py3.sif"
VENV_DIR="${PROJECT_DIR}/jamba_container"
OUTPUT_DIR="${PROJECT_DIR}/jamba-cipher-results"

cd $PROJECT_DIR
mkdir -p logs

# --- 2. DYNAMIC AUTO-CHAINING LOGIC ---
# If the final model exists, we've won. Stop the chain.
if [ -d "${OUTPUT_DIR}/final_model" ]; then
    echo "Check: '${OUTPUT_DIR}/final_model' found. Training is complete."
    echo "Breaking the chain. No more jobs will be submitted."
    exit 0
fi

# Submit the NEXT job immediately, set to start after this one ends (afterany)
echo "Relay active: Queuing the continuation job for after this run..."
sbatch --dependency=afterany:$SLURM_JOB_ID run_training.slurm || echo "Warning: sbatch failed, but training will continue."

# --- 3. PREPARE ENVIRONMENT ---
echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"

# NCCL/Distributed optimizations
export OMP_NUM_THREADS=8
MASTER_PORT=$((10000 + $RANDOM % 20000))

# --- 4. RUN TRAINING INSIDE SINGULARITY ---

singularity exec --nv \
    --pwd $PROJECT_DIR \
    -B /ceph/project/SW10-CausalLM:/ceph/project/SW10-CausalLM \
    ${CONTAINER} /bin/bash -c "
        set -x

        export PYTHONPATH='$PROJECT_DIR/src'
        export PYTHONUNBUFFERED=1
        export PYTHONNOUSERSITE=1
        
        source ${VENV_DIR}/bin/activate

        # Using torchrun to properly leverage all 4 allocated L4 GPUs
        torchrun --nproc_per_node=4 --master_port=$MASTER_PORT src/train.py "$@"
    "

echo "Job finished at $(date)"
