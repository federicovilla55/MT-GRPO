from comet import download_model, load_from_checkpoint
from .scorer import Scorer
import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np

class CometScorer(Scorer):
    def __init__(self, path_of_google_madlad, path_of_nllb, path_of_helsinki, device: str = "cuda"):
        super(CometScorer, self).__init__()
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        self.model = load_from_checkpoint(model_path)

        self.nllb_model = pipeline(task="translation", model=path_of_nllb, src_lang="eng_Latn", tgt_lang="ita_Latn", dtype=torch.float16, device='cuda', max_length=400)
        self.helsinki_transl = pipeline( "translation", model=path_of_helsinki, device='cuda', max_length=400, truncation=True)

        # self.madlad_tokenizer = AutoTokenizer.from_pretrained(path_of_google_madlad)
        # self.model_madlad = AutoModelForSeq2SeqLM.from_pretrained(
        #     path_of_google_madlad,
        #     device_map="cuda"
        # )

    def assign_score(self, src_text):
        result_nllb = self.nllb_model(src_text, batch_size=8, max_new_tokens=256)
        result_helsinki = self.helsinki_transl(src_text, batch_size=8, max_new_tokens=256)

        # texts_for_madlad = ["<2it> "+el for el in src_text]
        # inputs = self.madlad_tokenizer(
        #     texts_for_madlad,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True
        # ).to('cuda')
        # outputs = self.model_madlad.generate(
        #     **inputs,
        #     max_new_tokens=300
        # )

        # translations = self.madlad_tokenizer.batch_decode(
        #     outputs,
        #     skip_special_tokens=True
        # )

        result_nllb_values = [d["translation_text"] for d in result_nllb]
        result_helsinki_values =[d["translation_text"] for d in result_helsinki]
        
        num_rep = len(src_text)

        tot_translated = result_nllb_values+result_helsinki_values#+translations
        src_text_repeated = src_text+src_text#+src_text

        # data_nllb = [{ "src": origin_text, "mt": to_trans_text,}
        #     for origin_text, to_trans_text in zip(src_text, result_nllb_values)
        # ]
        # data_helsinki = [{ "src": origin_text, "mt": to_trans_text,}
        #     for origin_text, to_trans_text in zip(src_text, result_helsinki_values)
        # ]
        # data_madlad = [{ "src": origin_text, "mt": to_trans_text,}
        #     for origin_text, to_trans_text in zip(src_text, translations)
        # ]
        # model_output_nllb = self.model.predict(data_nllb, batch_size=8, gpus=1)
        # model_output_helsinki = self.model.predict(data_helsinki, batch_size=8, gpus=1)
        # model_output_madlad = self.model.predict(data_madlad, batch_size=8, gpus=1)

        data_score = [{ "src": origin_text, "mt": to_trans_text,}
            for origin_text, to_trans_text in zip(src_text_repeated, tot_translated)
        ]

        model_output= self.model.predict(data_score, batch_size=8, gpus=1)
        
        score_output = np.array(model_output.scores)
        score_output_reshape = score_output.reshape(2, num_rep)
        score_output_reshape = score_output_reshape.mean(axis = 0)
        score_output_reshape = score_output_reshape.tolist()
        return score_output_reshape
        
