from torch.utils.data import Dataset
from tqdm import tqdm
import json
from .dataset_grpo import DatasetGrpo
from .prompt import prompt_mod
from datasets import load_dataset

class Wmt19Dataset(DatasetGrpo):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        dataset = load_dataset("wmt/wmt19", "cs-en")
        dataset = dataset['validation']
        self.sentences = [el['translation']['en'] for el in dataset]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        string_el = self.sentences[index]
        messages = [
            {"role": "user", "content": prompt_mod(string_el)}
        ]
        text_prompt = super().process_to_return(messages)
        dict_to_return = {
          "prompt" : text_prompt,
        }
        return dict_to_return