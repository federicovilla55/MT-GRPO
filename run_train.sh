#!/bin/bash
#SBATCH --job-name=TestGRPO
#SBATCH --account=dphpc
#SBATCH --time=05:00:00
#SBATCH --output=output_%j.out
#SBATCH --error=output_%j.err

set -eo pipefail

echo "Running on $(hostname)"

#Use ${USER}

CMD="
/home/${USER}/DeepLearningProject/.venv/bin/python /home/${USER}/DeepLearningProject/train.py
"

srun bash -c "$CMD"

echo "Task finished"
