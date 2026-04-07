#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=modelnet40
#SBATCH --ntasks=1
#SBATCH --time=09:00:00
#SBATCH --output=modelnet40_output_%A.txt
#SBATCH --error=modelnet40_error_%A.txt

# Load necessary modules (adjust based on your environment)
module purge
module load 2023
module load CUDA/12.1.1

# my miniconda3 path
export PATH="$HOME/miniconda3/bin:$PATH"
unset -f conda 2>/dev/null      
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Activate the conda environment
conda activate r3pm

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT" || { echo "ERROR: cannot cd to REPO_ROOT=${REPO_ROOT}" >&2; exit 1; }
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "ERROR: REPO_ROOT=${REPO_ROOT} is not the r3pm_net tree (missing pyproject.toml)." >&2
  echo "Run: cd /path/to/r3pm_net && sbatch scripts/modelnet40.sh" >&2
  exit 1
fi

LOGDIR="${REPO_ROOT}/logs/slurm"
mkdir -p "$LOGDIR"
JOB_ID="${SLURM_JOB_ID:-local}"

# seeds=(42 61 92 114 123 456 789)
seeds=(42)

for seed in "${seeds[@]}"; do
  srun python scripts/eval_modelnet40.py --seed "${seed}" \
    >"${LOGDIR}/modelnet40_job${JOB_ID}_seed${seed}.log" 2>&1
done