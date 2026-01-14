import re
import random
import torch
import json
import nltk
import os

import numpy as np
import textdescriptives as td
import language_tool_python as tlp
import multiprocessing as mp
from vllm import LLM, SamplingParams

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch.utils.data import Subset
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

from dataset.dataset_wmt25 import Wmt25Dataset
from model.sentinel import SentinelScorer
# from model.comet import CometScorer
from evaluation_utils import (
    strip_reasoning,
    clean_string, 
    load_model,
    load_dataset,
    save_results,
    free_memory
)

MODEL_NAME="llama8b"
MODEL = "/work/scratch/gdemuri/llama8b/"
DATASET_NAME="wmt25"
BATCH_SIZE=8
N_SAMPLES=8
PATH_TO_SAVE=f"/home/gdemuri/dl-proj/results/"

PATH_MADLAD=""
PATH_NLLB=""
PATH_HELSINKI=""

def generate_responses(model_name, dataset_name, n_samples, use_vllm=False):
    """
    Generate responses for a given model. If model_name is null, returns original sentence as generated sentence.
    """
    print("Generate responses...")
    # if None, return original sentences as generated
    if model_name is None:
        _, tokenizer = load_model(model_name="meta-llama/Llama-3.2-3B-Instruct")
        dataset = load_dataset(dataset_name=dataset_name, tokenizer=tokenizer, n_samples=n_samples)
        results = [{"original": clean_string(strip_reasoning(sample)), "generated": clean_string(strip_reasoning(sample))} for sample in dataset]

    if not use_vllm: 
        # get model and dataset
        model, tokenizer = load_model(model_name=model_name)
        dataset = load_dataset(dataset_name=dataset_name, tokenizer=tokenizer, n_samples=N_SAMPLES, )

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
                        "original": clean_string(strip_reasoning(batch[i])),
                        "generated": decoded,
                    }
                )
    else:
        llm = LLM(
            model=model_name,
            dtype="half",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,  # lower if you OOM (0.65-0.75)
            max_model_len=2048,           # keep modest for low VRAM
            enforce_eager=False,
            max_num_seqs=1,
            # cpu_offload_gb=4,
            # swap_space=4,                 # CPU swap (GiB) if needed
            trust_remote_code=True,
        )

        sampling = SamplingParams(
            max_tokens=256,
            temperature=0.6,
            top_p=1.0,
        )

        # load dataset
        tokenizer = llm.get_tokenizer()
        dataset = load_dataset(dataset_name=dataset_name, tokenizer=tokenizer, n_samples=n_samples)

        # generate
        results = []
        batch_size = BATCH_SIZE
        print("Generating dataset...")
        for start in tqdm(range(0, len(dataset), batch_size)):
            batch = [dataset[i]["prompt"] for i in range(start, min(start + batch_size, len(dataset)))]

            # Apply chat template if available (important for instruct models)
            formatted_prompts = []
            for p in batch:
                formatted_prompts.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )

            outputs = llm.generate(formatted_prompts, sampling)

            for p, out in zip(batch, outputs):
                gen_text = out.outputs[0].text if out.outputs else ""
                gen_text = strip_reasoning(gen_text)

                results.append(
                    {
                        "original": clean_string(strip_reasoning(p)),
                        "generated": gen_text,
                    }
                )

    return results


def get_sentinel_score(samples):
    """
    Adds sentinel scores in-place:
      samples[i]["translation_difficulty"]["sentinel_score"]
    """
    print("Getting sentinel scores...")
    sentinel_model = SentinelScorer()

    generated = [s["generated"] for s in samples]

    g_scores = sentinel_model.assign_score(generated)["scores"]

    for i in range(len(samples)):
        samples[i]["translation_difficulty"] = {}
        samples[i]["translation_difficulty"]["sentinel_score"] = float(g_scores[i])

    return samples

def get_comet_score(samples):
    """
    Adds comet scores in-place:
        samples[i]["translation_difficulty"]["comet_score"]
    """
    print("Getting comet scores...")
    comet_model = CometScorer(PATH_MADLAD, PATH_HELSINKI, PATH_NLLB)

    generated = [s["generated"] for s in samples]

    g_scores = comet_model.assign_score(generated)

    for i in range(len(samples)):
        samples[i]["translation_difficulty"] = {}
        samples[i]["translation_difficulty"]["comet_score"] = float(g_scores[i])

    return samples

    
def get_grammatical_errors(samples):
    """Update JSON with grammatical errors"""
    print("Getting grammatical errors...")
    
    tool = tlp.LanguageTool('en-US')

    for sample in samples:
        text = sample.get("generated", "")
        matches = tool.check(text)
        sample["grammatical_errors"] = len(matches)

    return samples


def get_embeddings_score(samples):
    """
    Compute mean cosine similarity between all generated samples.
    Stores result in samples["embedding_similarity"].
    """
    print("Getting embeddings score...")
    generated = [s["generated"] for s in samples]

    sim_model = SentenceTransformer("all-MiniLM-L6-v2",device="cuda")

    # Clean and encode all generated texts
    clean_texts = [strip_reasoning(t) for t in generated]
    embeddings = sim_model.encode(
        clean_texts,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    # Pairwise cosine similarity matrix (N x N)
    sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)

    # Take upper triangle without diagonal
    n = sim_matrix.size(0)
    triu_indices = torch.triu_indices(n, n, offset=1)
    mean_similarity = sim_matrix[triu_indices[0], triu_indices[1]].mean().item()

    return mean_similarity


def get_complexity_scores(samples):
    """Get scores for complexity metrics"""
    print("Getting complexity scores...")
    generated = [s["generated"] for s in samples]

    # Get rix, entropy scores
    df = td.extract_metrics(text=generated, lang="en", metrics=["readability", "information_theory"])
    rix = df["rix"].to_numpy()
    entropy = df["entropy"].to_numpy()

    # Average word length (no stop words)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    stop_set = set(stopwords.words("english"))

    def avg_word_len(text):
        words = [t.lower() for t in word_tokenize(text)]
        words = [w for w in words if w not in stop_set]
        if not words:
            return 0.0
        return float(np.mean([len(w) for w in words]))

    # Average sentence length
    def avg_sentence_len_words(text):
        # split on .
        parts = [p.strip() for p in text.split(".") if p.strip()]
        if not parts:
            return 0.0
        lengths = []
        # for each part count words
        for p in parts:
            words = p.split()
            lengths.append(len(words))
        return float(np.mean(lengths)) if lengths else 0.0
    
    
    avg_word_lengths = [avg_word_len(t) for t in generated]
    avg_sentence_lengths = [avg_sentence_len_words(t) for t in generated]
    
    for i, s in enumerate(samples):
        s.setdefault("complexity_score", {})
        s["complexity_score"]["rix"] = float(rix[i])
        s["complexity_score"]["entropy"] = float(entropy[i])
        s["complexity_score"]["avg_word_length"] = avg_word_lengths[i]
        s["complexity_score"]["avg_sentence_length"] = avg_sentence_lengths[i]

    return samples

def get_judge_responses(samples):
    """
    Query a judge model (Llama 3.1 8B Instruct) via vLLM to score:
      - naturalness (0-100)
      - word rarity (0-100)
      - syntax complexity (0-100)
      - topics (list[str], 1-5)

    Writes results into:
      - samples["complexity_score"]["naturalness"]
      - samples["complexity_score"]["word_rarity"]
      - samples["complexity_score"]["syntax_complexity"]
      - samples["diversity_score"]["topics"]
    """
    print("Getting judge scores...")
    generated = [s["generated"] for s in samples]

    judge_model_name = "Qwen/Qwen3-4B"

    llm = LLM(
        model=judge_model_name,
        dtype="half",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,  # lower if you OOM (e.g., 0.70)
        max_model_len=2048,           # keep small for low VRAM
        enforce_eager=False,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
    )

    prompt_template = ("""
        Analyze the following text and return the answer in JSON. 
        We want to determine the following attributes:
        - naturalness: on a scale from 0 (wholy unnatural) to 100 (fully human-like and would occur in a corpus).
        - word rarity: on average, how rare are the words from 0 (average modern human would use this word daily) to 100 
        (average modern human would not understand the word).
        - syntax complexity: on a scale from 0 (simplest possible sentence) to 100 (most complex and hard to understand).
        - topics: list of 1 to 5 topics that the text is about. 
        Provide only the output in JSON and nothing else. 
        The output should look like this (no extra backticks or newlines): 
        {{ "naturalness": 80, "word rarity": 50, "syntax complexity": 70, "topics": ["science", "technology"] }} 
        The sentence to analyze is: > {SOURCE_TEXT}
        """
    )

    tok = llm.get_tokenizer()

    # Build prompts using the model's chat template when available
    prompts = []
    for text in generated:
        user_msg = prompt_template.format(SOURCE_TEXT=text)
        prompts.append(
            tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    outputs = llm.generate(prompts, sampling)

    def _extract_json_from_response(text):
        """Extract a JSON object from  model output"""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        candidate = m.group(0).strip()

        candidate = re.sub(r",\s*}", "}", candidate)
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _extract_value_from_json(d, key):
        v = d.get(key)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            m = re.search(r"-?\d+(\.\d+)?", v)
            return float(m.group(0)) if m else None
        return None

    for i, out in enumerate(outputs):
        samples[i].setdefault("complexity_score", {})
        samples[i].setdefault("diversity_score", {})

        text_out = out.outputs[0].text
        parsed = _extract_json_from_response(text_out)

        if not parsed:
            samples[i]["complexity_score"]["naturalness"] = None
            samples[i]["complexity_score"]["word_rarity"] = None
            samples[i]["complexity_score"]["syntax_complexity"] = None
            samples[i]["diversity_score"]["topics"] = None
            continue

        naturalness = _extract_value_from_json(parsed, "naturalness")
        word_rarity = _extract_value_from_json(parsed, "word rarity")
        syntax_complexity = _extract_value_from_json(parsed, "syntax complexity")

        topics = parsed.get("topics")
        if isinstance(topics, list):
            topics = [str(t) for t in topics][:5]
        else:
            topics = None

        samples["complexity_score"]["naturalness"][i] = naturalness
        samples["complexity_score"]["word_rarity"][i] = word_rarity
        samples["complexity_score"]["syntax_complexity"][i] = syntax_complexity
        samples["diversity_score"]["topics"][i] = topics

    return samples

def aggregate_all_results(samples):
    """Aggregate all results into a single dict"""
    agg = {}
    
    # ---- Helper to safely collect numeric values ----
    def collect(path):
        values = []
        for s in samples:
            d = s
            for k in path:
                d = d[k]
            values.append(d)
        return values
    
    # === Difficulty of translation metrics 
    translation_difficulty_metrics = [
        "sentinel_score",
        "comet_score",
    ]

    agg["translation_difficulty_score"] = {}
    for m in translation_difficulty_metrics:
        vals = collect(["translation_difficulty", m])
        agg["translation_difficulty_score"][m] = float(np.mean(vals))

    # === Complexity metrics 
    complexity_metrics = [
        "rix",
        "entropy",
        "avg_word_length",
        "avg_sentence_length",
        "naturalness",
        "word_rarity",
        "syntax_complexity",
    ]

    agg["complexity_score"] = {}
    for m in complexity_metrics:
        vals = collect(["complexity_score", m])
        agg["complexity_score"][m] = float(np.mean(vals))

    # === Grammatical errors 
    grammar_vals = collect(["grammatical_errors"])
    agg["grammatical_errors"] = float(np.mean(grammar_vals))

    # === Diversity
    # num topics
    all_topics = set()
    for s in samples:
        t = s["diversity_score"]["topics"]
        all_topics.update(t)
    num_topics = len(all_topics)
    
    # embeddings
    embedding_score = get_embeddings_score(samples)

    agg["diversity_scores"] = {
        "num_topics": num_topics,
        "embedding_score": embedding_score
    }
    return agg


def main():
    # === Obtain generations
    results = generate_responses(
        model_name=MODEL,
        dataset_name=DATASET_NAME,
        n_samples=N_SAMPLES,
        use_vllm=False
    )
    free_memory()

    # === Difficulty of translation
    # Sentinel 
    results = get_sentinel_score(samples=results)
    free_memory()
    
    # Comet
    # results = get_comet_score(samples=results)
    # free_memory()

    # === Grammatical errors
    results = get_grammatical_errors(samples=results)
    free_memory()

    # === Remaining complexity
    results = get_complexity_scores(samples=results)
    free_memory()

    # === LLM Judge: Complexity + Diversity of corpus 
    results = get_judge_responses(samples=results)
    free_memory()
    
    # === Save results
    save_results(results=results,
                 path_to_save=PATH_TO_SAVE)
    
    # === Aggregate results
    agg = aggregate_all_results(results=results, 
                          path_to_save=PATH_TO_SAVE + f"{MODEL_NAME}_{DATASET_NAME}_{N_SAMPLES}/{MODEL_NAME}_{DATASET_NAME}_{N_SAMPLES}.jsonl")
    
    # === Save aggregate results
    save_results(results=agg,
                 path_to_save=PATH_TO_SAVE + f"{MODEL_NAME}_{DATASET_NAME}_{N_SAMPLES}/aggregated.jsonl")



if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    
    main()
