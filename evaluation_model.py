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

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch.utils.data import Subset
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

from dataset.dataset_wmt25 import Wmt25Dataset
from model.sentinel import SentinelScorer
from model.comet import CometScorer
from evaluation_utils import (
    strip_reasoning,
    clean_string, 
    load_model,
    load_dataset,
    save_results,
    free_memory
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME="llama_sentinel_8b_temp=0.9_top_p=0.7_n_sample=100_num_gen=10"
# MODEL = "/cluster/scratch/arsood/DeepLearningProject/outputModels/grpo_qwen4b_comet_v2/checkpoint-6174"
# MODEL = "/cluster/scratch/arsood/best_qwen_4b_dl/artifacts/qwen4b-grpo-checkpoint-386-2hy4ani0:v0"
# MODEL = "/cluster/scratch/arsood/qwen_4b_off"
MODEL = os.path.join(BASE_DIR, "best_llama_8b")
# MODEL = "/cluster/scratch/arsood/DeepLearningProject/outputModels/grpo_qwen1b_comet_v3/checkpoint-3087"
MODEL_JUDGE = os.path.join(BASE_DIR, "qwen_4b_off")
DATASET_NAME="wmt19"
BATCH_SIZE=50
N_SAMPLES=100
NUM_GENERATION = 10
PATH_TO_SAVE = os.path.join(BASE_DIR, "results")

PATH_MADLAD = os.path.join(BASE_DIR, "madlad-google")
PATH_NLLB = os.path.join(BASE_DIR, "nllb-200")
PATH_HELSINKI = os.path.join(BASE_DIR, "helsinki-nlp")

def generate_responses(model_name, dataset_name, n_samples, use_vllm=False):
    """
    Generate responses for a given model. If model_name is null, returns original sentence as generated sentence.
    """
    print("Generate responses...")

    if not use_vllm: 
        # get model and dataset
        model, tokenizer = load_model(model_name=model_name)
        dataset = load_dataset(dataset_name=dataset_name, tokenizer=tokenizer, n_samples=N_SAMPLES)

        # generate responses
        results = []
        batch_size = BATCH_SIZE
        print("Generating dataset...")
        for start in tqdm(range(0, len(dataset), batch_size)):
            batch = [dataset[i]["prompt"] for i in range(start, min(start + batch_size, len(dataset)))]
            for num_generation in range(NUM_GENERATION):
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
                prompt_width  = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.9,
                        do_sample=True,
                        top_p=0.7,
                    )

                for i in range(len(batch)):
                    gen_tokens = outs[i, prompt_width:]
                    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    decoded = strip_reasoning(decoded)
                    # pattern = r"\s*<\|im_end\|>\s*\n<\|im_start\|>assistant\s*$"
                    pattern = r"\s*<\|eot_id\|>\s*<\|start_header_id\|>assistant<\|end_header_id\|>\s*"
                    results.append(
                        {
                            "original": re.sub(pattern, "", clean_string(strip_reasoning(batch[i]))),
                            "generated": decoded,
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
        # samples[i]["translation_difficulty"] = {}
        samples[i]["translation_difficulty"]["comet_score"] = float(g_scores[i])

    return samples

def get_sentinel_score_original(samples):
    """
    Adds sentinel scores in-place:
      samples[i]["translation_difficulty"]["sentinel_score"]
    """
    print("Getting sentinel scores...")
    sentinel_model = SentinelScorer()

    generated = [s["original"] for s in samples]

    g_scores = sentinel_model.assign_score(generated)["scores"]

    for i in range(len(samples)):
        samples[i]["translation_difficulty_original"] = {}
        samples[i]["translation_difficulty_original"]["sentinel_score"] = float(g_scores[i])

    return samples

def get_comet_score_original(samples):
    """
    Adds comet scores in-place:
        samples[i]["translation_difficulty"]["comet_score"]
    """
    print("Getting comet scores...")
    comet_model = CometScorer(PATH_MADLAD, PATH_HELSINKI, PATH_NLLB)

    generated = [s["original"] for s in samples]

    g_scores = comet_model.assign_score(generated)

    for i in range(len(samples)):
        # samples[i]["translation_difficulty"] = {}
        samples[i]["translation_difficulty_original"]["comet_score"] = float(g_scores[i])

    return samples


    
def get_grammatical_errors(samples):
    """Update JSON with grammatical errors"""
    print("Getting grammatical errors...")
    
    tool = tlp.LanguageTool(
            'en-US',
            remote_server='http://127.0.0.1:8085'
        )

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

def get_judge_responses(samples, model_name, use_vllm = False):
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

    if not use_vllm:
        model, tokenizer = load_model(model_name=model_name)

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

    prompts = []
    for text in generated:
        user_msg = prompt_template.format(SOURCE_TEXT=text)
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    batch_size = 50

    all_outputs = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
            )

        # trim generation (same logic you already use)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]

        decoded = tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )

        all_outputs.extend(decoded)
    outputs = all_outputs
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

        text_out = out#.outputs[0].text
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

        samples[i]["complexity_score"]["naturalness"] = naturalness
        samples[i]["complexity_score"]["word_rarity"] = word_rarity
        samples[i]["complexity_score"]["syntax_complexity"] = syntax_complexity
        samples[i]["diversity_score"]["topics"] = topics

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

    agg["translation_difficulty_original_score"] = {}
    for m in translation_difficulty_metrics:
        vals = collect(["translation_difficulty_original", m])
        agg["translation_difficulty_original_score"][m] = float(np.mean(vals))

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
    results = get_comet_score(samples=results)
    free_memory()

    results = get_sentinel_score_original(samples=results)
    free_memory()
    
    # Comet
    results = get_comet_score_original(samples=results)
    free_memory()

    # === Grammatical errors
    results = get_grammatical_errors(samples=results)
    free_memory()

    # === Remaining complexity
    results = get_complexity_scores(samples=results)
    free_memory()

    # === LLM Judge: Complexity + Diversity of corpus 
    results = get_judge_responses(samples=results, model_name = MODEL_JUDGE)
    free_memory()
    # # === Save results
    # save_results(results=results,
    #              path_to_save=PATH_TO_SAVE)
    
    # === Aggregate results
    # agg = aggregate_all_results(results)
    
    # === Save aggregate results
    path_to_save_file = PATH_TO_SAVE + f"{MODEL_NAME}_{DATASET_NAME}_{N_SAMPLES}/aggregated.jsonl"
    save_results(results=results,
                 path_to_save=path_to_save_file)



if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    
    main()
