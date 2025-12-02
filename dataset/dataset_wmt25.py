from torch.utils.data import Dataset
from tqdm import tqdm
import json
from .dataset_grpo import DatasetGrpo
from .prompt import prompt_mod

class Wmt25Dataset(DatasetGrpo):
    def __init__(self, tokenizer, lan_source = 'en', lan_target = 'en'):
        super().__init__(tokenizer)
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
        messages = [
            {"role": "user", "content": prompt_mod(string_el)}
        ]
        text_prompt = super().process_to_return(messages)
        dict_to_return = {
          "prompt" : text_prompt,
        }
        return dict_to_return