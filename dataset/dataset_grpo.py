from torch.utils.data import Dataset


class DatasetGrpo(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    
    def process_to_return(self, prompt):
        text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return text