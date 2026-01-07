import re
import random
import torch
import json

import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch.utils.data import Subset
from tqdm import tqdm

from dataset.dataset_wmt25 import Wmt25Dataset
from model.sentinel import SentinelScorer

MODEL_NAME = "/work/scratch/gdemuri/qwen4b/"
DATASET_NAME="wmt25"
BATCH_SIZE=8
N_SAMPLES=100
PATH_TO_SAVE=f"/home/gdemuri/dl-proj/results/qwen4b_{DATASET_NAME}_{N_SAMPLES}.jsonl"

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


def generate_dataset():
    # get model 
    model, tokenizer = load_model(model_name=MODEL_NAME)

    # get dataset for generation
    dataset = load_dataset(dataset_name=DATASET_NAME, tokenizer=tokenizer, n_samples=N_SAMPLES, )

    # generate responses
    results = []
    batch_size = BATCH_SIZE
    print("Generating dataset...")
    for start in tqdm(range(0, len(dataset), batch_size)):
        batch = [dataset[i]["prompt"] for i in range(start, min(start + batch_size, len(dataset)))]

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        prompt_width  = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.6,
                do_sample=True,
            )

        for i in range(len(batch)):
            gen_tokens = outs[i, prompt_width:]
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            decoded = strip_reasoning(decoded)

            results.append(
                {
                    "original": {"sentence": clean_string(strip_reasoning(batch[i]))},
                    "generated": {"sentence": decoded},
                }
            )

    return results


def evaluate_with_sentinel(samples):
    """
    Adds sentinel scores in-place:
      samples[i]["original"]["sentinel_original"]
      samples[i]["generated"]["sentinel_generated"]
    """
    print("Getting sentinel scores...")
    sentinel_model = SentinelScorer()

    originals = [s["original"]["sentence"] for s in samples]
    generated = [s["generated"]["sentence"] for s in samples]

    o_scores = sentinel_model.assign_score(originals)["scores"]
    g_scores = sentinel_model.assign_score(generated)["scores"]

    for i in range(len(samples)):
        samples[i]["original"]["sentinel_score"] = float(o_scores[i])
        samples[i]["generated"]["sentinel_score"] = float(g_scores[i])

    return samples


def get_translations(samples, nllb_model_name="facebook/nllb-200-distilled-600M"):
    """
    Adds Italian translations in-place:
      samples[i]["original"]["translation"]
      samples[i]["generated"]["translation"]
    """
    print("Getting translations...")
    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
    tokenizer.src_lang = "eng_Latn"
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("ita_Latn")

    # get model
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name).to("cuda")
    translation_model.eval()

    originals = [s["original"]["sentence"] for s in samples]
    generated = [s["generated"]["sentence"] for s in samples]

    def translate(texts):
        batch_size = BATCH_SIZE
        out = []
        for start in tqdm(range(0, len(texts), batch_size)):
            chunk = texts[start:start + batch_size]
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(translation_model.device)
            with torch.no_grad():
                gen = translation_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                )
            out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        return out

    o_it = translate(originals)
    g_it = translate(generated)

    for i in range(len(samples)):
        samples[i]["original"]["translation"] = o_it[i]
        samples[i]["generated"]["translation"] = g_it[i]

    return samples

def save_results(results, path_to_save):

    with open(path_to_save, "w", encoding="utf-8") as f:
        for item in results:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True
                ) + "\n"
            )
    
def main():
    results = generate_dataset()

    results = evaluate_with_sentinel(samples=results)

    results = get_translations(samples=results)

    save_results(results=results,
                 path_to_save=PATH_TO_SAVE)


if __name__=="__main__":
    main()