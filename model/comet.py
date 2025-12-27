from comet import download_model, load_from_checkpoint
from .scorer import Scorer
import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

class CometScorer(Scorer):
    def __init__(self, device: str = "cuda"):
        super(CometScorer, self).__init__()
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        self.model = load_from_checkpoint(model_path)
        self.nllb_model = pipeline(task="translation", model="facebook/nllb-200-distilled-1.3B", src_lang="eng_Latn", tgt_lang="fra_Latn", dtype=torch.float16, device='cuda')
        path_of_model = "/users/fvilla/scratch/DeepLearningProject/qwen_4b"
        self.tokenizer = AutoTokenizer.from_pretrained(path_of_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            path_of_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def assign_score(self, src_text, tran_text):
        
        data = [
            {
            "src": origin_text,
            "mt": to_trans_text,
            }
            for origin_text, to_trans_text in zip(src_text, tran_text)
        ]
        model_output = self.model.predict(data, batch_size=8, gpus=1)
        print(self.nllb_model(src_text))
        return model_output.scores
        
