import wandb
import os

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)
wandb.init(project="grpo_training")

artifact = wandb.Artifact(
    name="my-model",
    type="model",
    description="Saved model directory"
)

artifact.add_dir("/cluster/scratch/arsood/DeepLearningProject/outputModels/grpo_qwen4b_comet_v2/checkpoint-6174")
wandb.log_artifact(artifact)

wandb.finish()
