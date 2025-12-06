from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from torch import nn 
import torch
import random
from torch.utils.data import DataLoader, ConcatDataset
import re
import os
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.dataset_tatoeba import TatoebaDataset
from dataset.dataset_wmt25 import Wmt25Dataset
from tqdm import tqdm
from model.sentinel import SentinelScorer


sentinel_model = SentinelScorer()

def reward_grpo(completions, **kwargs):
    #rewards_to_give = [0.8] * len(completions)
    rewards_to_give = sentinel_model.assign_score(completions)
    reward_to_give = 1-np.array(rewards_to_give["scores"])
    return reward_to_give
    #return rewards_to_give

path_of_model = f"/home/{os.getenv('USER')}/DeepLearningProject/qwen_1b"
path_of_model = "/work/scratch/fvilla/grpo_qwen1b_sentinel/checkpoint-6_old"
tokenizer = AutoTokenizer.from_pretrained(path_of_model)
model = AutoModelForCausalLM.from_pretrained(
    path_of_model,
    torch_dtype="auto",
    device_map="auto"
)

for param in model.parameters():
    param.requires_grad = True

dataset_tatoeba = TatoebaDataset(tokenizer=tokenizer, filtered=True, k=350)
dataset_wmt25 = Wmt25Dataset(tokenizer=tokenizer)
concat_data = ConcatDataset([dataset_tatoeba])


#print(len(concat_data))
# dataloader = DataLoader(concat_data, shuffle = True, batch_size = 1)


training_args = GRPOConfig(
    learning_rate=2e-5,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    max_completion_length=512,
    num_generations=4,
    max_prompt_length=1024,
    fp16=False,
    output_dir=f"/work/scratch/{os.getenv('USER')}/grpo_qwen1b_sentinel",                        
    logging_steps=1,
    temperature = 0.7,
    top_p=0.9,
    use_liger_loss=True,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_grpo,
    args=training_args,
    train_dataset=concat_data,
    processing_class = tokenizer,
)
trainer.train()