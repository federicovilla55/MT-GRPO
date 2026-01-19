from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from comet import download_model, load_from_checkpoint
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

local_model = os.path.join(BASE_DIR, "Qwen3-4B-Instruct-2507")
tokenizer = AutoTokenizer.from_pretrained(local_model)
model = AutoModelForCausalLM.from_pretrained(
    local_model,
    torch_dtype=torch.float16,
    device_map="cuda"
)

prompt = "Make this sentence more difficult, such that translation model have more difficulty to translate it(directly output the difficult sentence, do not make it to long close to original lenght): I will play in a volleyball tournament tomorrow against the strongest team in the city."

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

matrix_of_completition = []
for i in range(10):
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=1,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    matrix_of_completition.append(content)
    

data = [ 
  {"src" : el} for el in matrix_of_completition
]

from  sentinel.sentinel_metric import download_model, load_from_checkpoint

model_path = download_model("Prosho/sentinel-src-25")
model = load_from_checkpoint(model_path)

output = model.predict(data, batch_size=8, gpus=1)
print(output)

for i, (out, d) in enumerate(zip(output["scores"],data)):
    print(i, ", score: ", out, ", prompt: ", d)
