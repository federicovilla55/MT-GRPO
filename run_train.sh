#!/bin/bash
#SBATCH --job-name=TrainGRPO
#SBATCH --account=dl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err

set -eo pipefail

echo "Running on $(hostname)"

PROJECT_PATH=$(cd "$(dirname "$0")"; pwd)
source "$PROJECT_PATH/.venv/bin/activate"

CMD="
python \"$PROJECT_PATH/train.py\"
"

srun bash -c "$CMD"

echo "Task finished"