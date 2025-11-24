from torch.utils.data import Dataset
from tqdm import tqdm
import json
import pandas as pd

class Tatoeba(Dataset):
    def __init__(self):
        super().__init__()
        df = pd.read_csv("data/eng_sentences.tsv", sep="\t", header=None, names=["id", "lang", "text"])
        self.sentences = df["text"].tolist()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        string_el = self.sentences[index]
        return string_el