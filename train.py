from torch.utils.data import ConcatDataset
import re
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.dataset_tatoeba import TatoebaDataset
from dataset.dataset_wmt25 import Wmt25Dataset
from sentence_transformers import SentenceTransformer, util
from model.sentinel import SentinelScorer
import textdescriptives as td
import language_tool_python

tool = language_tool_python.LanguageTool(
    'en-US',
    remote_server='http://127.0.0.1:8085'
)


WANDB_API_KEY = os.getenv("WANDB_API_KEY")
report_to = "none"

if WANDB_API_KEY:
    import wandb
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project="grpo_training")
    report_to = ["wandb"]

torch.set_float32_matmul_precision('high')

device = "cuda" if torch.cuda.is_available() else "cpu"

sentinel_model = SentinelScorer()


def strip_reasoning(text):
    # Remove the <think>...</think> tags and their content, which is not important for scoring and translation purposes.
    clean_text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("assistant", "").strip()
    clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
    clean_text = clean_text.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("assistant", "").strip()
    return clean_text

def extract_original_sentence(prompt):
    delimiter = "Sentence you have to change: "
    if delimiter in prompt:
        original_sentence = prompt.split(delimiter)[-1].strip()
    else:
        original_sentence = prompt
    return original_sentence

def reward_sentinel(prompts, completions, **kwargs):
    clean_completions = [strip_reasoning(text) for text in completions]
    
    sentinel_scores = -np.array(sentinel_model.assign_score(clean_completions)["scores"])
    
    df = td.extract_metrics(text=clean_completions, lang="en", metrics=["readability", "information_theory"])
    rix = df["rix"].values
    entropy = df["entropy"].values

    rewards = []
    for i in range(len(clean_completions)):
        score = sentinel_scores[i] / 3.0
        
        rewards.append(score)
        
    return rewards

def reward_comet(completions, **kwargs):
    clean_completions = [strip_reasoning(text) for text in completions]
    rewards_to_give = comet_model.assign_score(clean_completions)
    reward_to_give = 1-np.array(rewards_to_give)
    return (reward_to_give).tolist()

def reward_avoid_illegal(prompts, completions, **kwargs):
    # Penalize completions that exceed 256 tokens or contain non ascii characters
    rewards = []
    penalty_value = -5.0

    for completion in completions:
        current_penalty = 0.0
        clean_completion = strip_reasoning(completion)

        if len(tokenizer.encode(clean_completion)) >= 256:
            current_penalty += penalty_value

        if not clean_completion.isascii():
            current_penalty += penalty_value
            
        rewards.append(current_penalty)

    return rewards

def reward_grammatical_correctness(prompts, completions, **kwargs):
    """
    Rewards the model for grammatical correctness using language_tool_python.
    Returns higher rewards for text with fewer grammatical errors.
    """
    rewards = []
    
    for completion in completions:
        clean_completion = strip_reasoning(completion)
        
        if not clean_completion:
            rewards.append(-10.0)
            continue
            
        try:
            matches = tool.check(clean_completion)
            
            error_count = len(matches)
            
            if error_count == 0:
                reward_score = 1.0  # Reward perfect grammar
            elif error_count <= 2:
                reward_score = -0.5
            else:
                reward_score = max(-20, -2.0 * error_count)
            
            rewards.append(reward_score)
            
        except Exception as e:
            rewards.append(0.0)
    
    return rewards

def reward_relative_length(prompts, completions, **kwargs):
    # Reward based on the relative length of the generated sentence compared to the original sentence.
    rewards = []
    
    # Printing the generated phrase every 10 steps
    if not hasattr(reward_relative_length, "step"):
        reward_relative_length.step = 0

    reward_relative_length.step += 1
    should_print = (reward_relative_length.step % 5 == 0)
    
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        clean_source = extract_original_sentence(prompt)
        clean_completion = strip_reasoning(completion)

        len_src = max(1, len(clean_source.split()))
        len_gen = len(clean_completion.split())
        
        ratio = len_gen / len_src
        
        if 0.8 <= ratio <= 1.4:
            rewards.append(1.0)  # Reward good length ratio
        elif 1.4 < ratio <= 1.8:
            rewards.append(0.2)
        elif ratio > 1.8:
            penalty = max(-10, -3.0 * (ratio - 1.8))
            rewards.append(penalty)
        else:
            rewards.append(-5.0)

        if i == 0 and should_print:
            # Print generated content every 10 steps
            print(f"{'='*20}")
            print("Original sentence: ", clean_source)
            print("Generated sentence: ", clean_completion)
            print(f"{'='*20}")
        
    return rewards

sim_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def reward_semantic_similarity(prompts, completions, **kwargs):
    # Reward based on semantic similarity between  and generated sentence.
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        source = extract_original_sentence(prompt)
        emb1 = sim_model.encode(source, convert_to_tensor=True, show_progress_bar=False)
        clean_comp = strip_reasoning(completion)
        emb2 = sim_model.encode(clean_comp, convert_to_tensor=True, show_progress_bar=False)
        
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        
        if 0.5 <= similarity <= 0.90:
            rewards.append(1.0)  # Reward good semantic similarity range
        # Enforce a minimum difference:
        elif similarity > 0.90:
            rewards.append(-4.0 * (similarity - 0.90)) 
        else:
            rewards.append(-10.0) 
            
    return rewards




BASE_DIR = os.path.dirname(os.path.abspath(__file__))

path_of_model = os.path.join(BASE_DIR, "model", "qwen_1_b")

tokenizer = AutoTokenizer.from_pretrained(path_of_model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 


model = AutoModelForCausalLM.from_pretrained(
    path_of_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

for param in model.parameters():
    param.requires_grad = True

k = 198 # Number of minimum characters for phrases to be considered during training. 
num_generations = 8 # Number of generations per prompt. 
batch_size = 16 # Batch size for training. 
grad_accum_steps = 1 # Gradient accumulation steps. 
repetition_penalty = 1.4 # Repetition penalty. 
temperature = 0.7 # Temperature for generation. 
learning_rate = 1e-5 # Training Learning rate. 

dataset_tatoeba = TatoebaDataset(tokenizer=tokenizer, filtered=True, k=k)
dataset_wmt25 = Wmt25Dataset(tokenizer=tokenizer)
concat_data = ConcatDataset([dataset_tatoeba])

print("Starting training...")
print("Parameters: ")
print(f"Learning rate: {learning_rate}")
print(f"Number of epochs: 1")
print(f"Batch size: {batch_size}")
print(f"Number of generations: {num_generations}")
print(f"Gradient accumulation steps: {grad_accum_steps}")
print(f"Minimum number of characters per considered phrase: {k}")
print(f"Repetition penalty: {repetition_penalty}")
print(f"Temperature: {temperature}")

training_args = GRPOConfig(
    learning_rate=learning_rate,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum_steps,
    max_completion_length=256,
    num_generations=num_generations,
    max_prompt_length=512,
    fp16=False,
    bf16=True,
    output_dir=os.path.join(BASE_DIR, "outputModels", "grpo_qwen1b_comet"),                        
    logging_steps=1,
    repetition_penalty=repetition_penalty,
    temperature=temperature,
    top_p=0.9,
    report_to=report_to,
    run_name="grpo_qwen1b_comet",
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    reward_weights=[
        2,      # sentinel / comet
        1.5,    # relative_length
        2,      # avoid illegal characters or too long
        1.5,    # semantic_similarity
        1.5,    # grammatical_correctness
    ],
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        reward_sentinel,
        reward_relative_length, 
        reward_avoid_illegal,
        reward_semantic_similarity,
        reward_grammatical_correctness
    ],

    args=training_args,
    train_dataset=concat_data,
    processing_class = tokenizer,
)
trainer.train()

print("Training completed.")