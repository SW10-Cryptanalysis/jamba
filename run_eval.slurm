#!/bin/bash
#SBATCH --job-name=jamba_eval
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l4:4

set -e

# --- 1. DEFINE PATHS ---
PROJECT_DIR="/ceph/project/SW10-CausalLM/jamba"
CONTAINER="${PROJECT_DIR}/pytorch_25.03-py3.sif"
VENV_DIR="${PROJECT_DIR}/jamba_container"
OUTPUT_DIR="${PROJECT_DIR}/jamba-cipher-results"

cd $PROJECT_DIR
mkdir -p logs

# --- 2. PREPARE ENVIRONMENT ---
echo "Starting evaluation job ${SLURM_JOB_ID} on $(hostname) at $(date)"

# Ensure the model exists before spinning up the container
if [ ! -d "${OUTPUT_DIR}/final_model" ]; then
    echo "❌ Error: '${OUTPUT_DIR}/final_model' not found."
    echo "Please ensure training has completed successfully."
    exit 1
fi

export OMP_NUM_THREADS=8

# --- 3. RUN EVALUATION INSIDE SINGULARITY ---
singularity exec --nv \
    --pwd $PROJECT_DIR \
    -B /ceph/project/SW10-CausalLM:/ceph/project/SW10-CausalLM \
    -B ${VENV_DIR}:/scratch/venv \
    ${CONTAINER} /bin/bash -c "
        export PYTHONPATH=''
        export PYTHONNOUSERSITE=1
        source /scratch/venv/bin/activate

        # Standard Python run; device_map='auto' in eval.py will handle the 4 GPUs
        python src/eval.py
    " | tee -a logs/eval_live_${SLURM_JOB_ID}.log

echo "Evaluation finished at $(date)"
