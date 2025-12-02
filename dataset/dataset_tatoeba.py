from torch.utils.data import Dataset
from tqdm import tqdm
import json
import pandas as pd
from .dataset_grpo import DatasetGrpo
from .prompt import prompt_mod

class TatoebaDataset(DatasetGrpo):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        df = pd.read_csv("data/eng_sentences.tsv", sep="\t", header=None, names=["id", "lang", "text"])
        self.sentences = df["text"].tolist()

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
    
    