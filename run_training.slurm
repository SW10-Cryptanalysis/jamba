#!/bin/bash
#SBATCH --job-name=jamba_cipher
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:l4:1

set -x
set -e

# --- 1. DEFINE PATHS & DYNAMIC OUTPUT ---
PROJECT_DIR="/ceph/project/SW10-CausalLM/jamba"
CONTAINER="${PROJECT_DIR}/pytorch_25.03-py3.sif"
VENV_DIR="${PROJECT_DIR}/jamba_container"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Virtual environment not found. Building it now with system packages..."
    singularity exec --nv \
        --pwd $PROJECT_DIR \
        -B /ceph/project/SW10-CausalLM:/ceph/project/SW10-CausalLM \
        ${CONTAINER} /bin/bash -c "
            # Fail fast: exit immediately if any command inside the container fails
            set -e 
            
            echo 'Creating virtual environment using built-in venv...'
            python -m venv ${VENV_DIR} --system-site-packages
            
            echo 'Activating environment...'
            source ${VENV_DIR}/bin/activate
            
            echo 'Installing uv into the virtual environment...'
            python -m pip install uv
            
            echo 'Syncing dependencies via uv...'
            uv sync --active
        "
else
    echo "Virtual environment already exists. Skipping build."
fi

# Check if "--spaced" was passed in the arguments to the SLURM script
if [[ "$*" == *"--spaced"* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/jamba-cipher-results-spaced"
else
    OUTPUT_DIR="${PROJECT_DIR}/jamba-cipher-results"
fi

cd $PROJECT_DIR
mkdir -p logs

# --- 2. DYNAMIC AUTO-CHAINING LOGIC ---
if [ -d "${OUTPUT_DIR}/final_model" ]; then
    echo "Check: '${OUTPUT_DIR}/final_model' found. Training is complete."
    echo "Breaking the chain. No more jobs will be submitted."
    exit 0
fi

echo "Relay active: Queuing the continuation job for after this run..."
sbatch --dependency=afterany:$SLURM_JOB_ID run_training.slurm "$@" || echo "Warning: sbatch failed, but training will continue."

# --- 3. PREPARE ENVIRONMENT ---
echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date) with output at ${OUTPUT_DIR}"

export OMP_NUM_THREADS=8
MASTER_PORT=$((10000 + $RANDOM % 20000))

# --- 4. RUN TRAINING ---
singularity exec --nv \
    --pwd $PROJECT_DIR \
    -B /ceph/project/SW10-CausalLM:/ceph/project/SW10-CausalLM \
    ${CONTAINER} /bin/bash -c "
        set -x
        export PYTHONPATH='$PROJECT_DIR/src'
        export PYTHONUNBUFFERED=1
        export PYTHONNOUSERSITE=1
        
        source ${VENV_DIR}/bin/activate

        torchrun --nproc_per_node=4 --master_port=$MASTER_PORT src/train.py "$@"
    "

echo "Job finished at $(date)"
