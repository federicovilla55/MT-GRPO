from torch.utils.data import Dataset
from tqdm import tqdm
import json

class Wmt25Dataset(Dataset):
    def __init__(self, lan_source = 'en', lan_target = 'en'):
        super().__init__()

        all_data = []
        all_keys = set()
        with open("data/wmt25-genmt-humeval.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():                  
                    obj = json.loads(line)        
                    all_data.append(obj)
                    all_keys.update(obj.keys()) 
    
        self.sentences = []

        for el in tqdm(all_data):
            if(el["doc_id"][0:2] == lan_source):
                self.sentences.append(el["src_text"])
            if(el["doc_id"][3:5] == lan_target):
                self.sentences.append(el["tgt_text"])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        string_el = self.sentences[index]
        return string_el