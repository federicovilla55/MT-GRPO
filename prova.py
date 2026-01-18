from datasets import load_dataset
from dataset.dataset_wmt19 import Wmt19Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/cluster/scratch/arsood/DeepLearningProject/outputModels/grpo_qwen1b_comet_v3/checkpoint-3087"
tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
dataset = Wmt19Dataset(tokenizer)

for el in dataset:
    print(el)
    input_l = input("Ciao?")