#!/bin/bash
#SBATCH --job-name=TestGRPO
#SBATCH --account=large-sc-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err

## Commented: #SBATCH --partition=mi300
## Commented: #SBATCH --environment=/users/fvilla/scratch/assignment-2/amd_env.toml

set -eo pipefail

echo "Running on $(hostname)"

# Set wandb api key:
export WANDB_API_KEY="295e5fe387c47458635e18d72fb1e3eb7b9e5235"
export WANDB_SILENT=true

#Use ${USER}
export TOKENIZERS_PARALLELISM=false

CMD="
/users/fvilla/scratch/DeepLearningProject/.venv/bin/python /users/fvilla/scratch/DeepLearningProject/train.py
"

# .venv/bin/python train.py

srun bash -c "$CMD"

echo "Task finished"