import re
import random
import torch
import json
import nltk

import numpy as np
import textdescriptives as td
import language_tool_python as tlp

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Subset

from dataset.dataset_wmt25 import Wmt25Dataset
from dataset.dataset_tatoeba import TatoebaDataset
from dataset.dataset_wmt19 import Wmt19Dataset
import os

def strip_reasoning(text):
    clean_text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("assistant", "").strip()
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return clean_text

def clean_string(s):
    marker = "Sentence you have to change:"
    if marker in s:
        return s.split(marker, 1)[1].lstrip()
    
    return s

def load_model(model_name):
    """Helper function to load model and tokenizer"""

    print(f"Loading model {model_name}...")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" 

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    return model, tokenizer

def load_dataset(dataset_name, tokenizer, n_samples=-1, min_tokens=None, max_tokens=None, seed=42):
    """Load dataset, randomly subsamples it and filters it"""
    
    # load dataset
    if dataset_name == "wmt25":
        dataset = Wmt25Dataset(tokenizer=tokenizer)
    elif dataset_name == "tatoeba":
        dataset = TatoebaDataset(tokenizer=tokenizer, filtered=True, k=198)
    elif dataset_name == "wmt19":
        dataset = Wmt19Dataset(tokenizer=tokenizer)
    else:
        raise ValueError(f"Please specify a supported dataset to load")

    # filter it
    indices = []
    for i in range(len(dataset)):
        text = dataset[i]["prompt"]
        tok_len = len(tokenizer.encode(text))

        if min_tokens is not None and tok_len < min_tokens:
            continue
        if max_tokens is not None and tok_len > max_tokens:
            continue

        indices.append(i)
    print(f"Samples left after filtering: {len(indices)}")

    # subsample it
    if n_samples == -1:
        sampled_dataset = dataset
    else:
        rng = random.Random(seed)
        sampled_indices = rng.sample(indices, n_samples)
        sampled_dataset = Subset(dataset, sampled_indices)

    return sampled_dataset

def save_results(results, path_to_save):
    """Save results file to json"""
    save_dir = os.path.dirname(path_to_save)
    os.makedirs(save_dir, exist_ok=True)
    print(path_to_save)
    with open(path_to_save, "w", encoding="utf-8") as f:
        for item in results:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True
                ) + "\n"
            )

def free_memory():
    import gc
    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()