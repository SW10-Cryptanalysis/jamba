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

# --- 1. DEFINE PATHS & DYNAMIC FOLDER LOGIC ---
PROJECT_DIR="/ceph/project/SW10-CausalLM/jamba"
CONTAINER="${PROJECT_DIR}/pytorch_25.03-py3.sif"
VENV_DIR="${PROJECT_DIR}/jamba_container"

# Check for the --spaced flag in the arguments passed to sbatch
if [[ "$*" == *"--spaced"* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/jamba-cipher-results-spaced"
    SPACED_FLAG="--spaced"
else
    OUTPUT_DIR="${PROJECT_DIR}/jamba-cipher-results"
    SPACED_FLAG=""
fi

MODEL_PATH="${OUTPUT_DIR}/final_model"

cd $PROJECT_DIR
mkdir -p logs

# --- 2. PREPARE ENVIRONMENT ---
echo "Starting evaluation job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Targeting model at: ${MODEL_PATH}"

# Ensure the model exists before spinning up the container
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: '${MODEL_PATH}' not found."
    echo "Please ensure training has completed for this specific mode."
    exit 1
fi

export OMP_NUM_THREADS=8

# --- 3. RUN EVALUATION INSIDE SINGULARITY ---
# We pass the path and the flag directly to eval.py
singularity exec --nv \
    --pwd $PROJECT_DIR \
    -B /ceph/project/SW10-CausalLM:/ceph/project/SW10-CausalLM \
    ${CONTAINER} /bin/bash -c "
        export PYTHONPATH='$PROJECT_DIR/src'
        export PYTHONNOUSERSITE=1
        source ${VENV_DIR}/bin/activate

        python src/eval.py --model_path ${MODEL_PATH} ${SPACED_FLAG}
    " | tee -a logs/eval_live_${SLURM_JOB_ID}.log

echo "Evaluation finished at $(date)"