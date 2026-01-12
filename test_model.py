import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os
from dataset.prompt import prompt_mod
from model.sentinel import SentinelScorer
import textdescriptives as td

sentinel = SentinelScorer()


MODEL_PATH = "outputModels/grpo_llama8b_sentinel_v2/checkpoint-386"
print(f"Loading model from: {MODEL_PATH}")

TEST_PHRASE = """During the first technical interview, we'll focus on coding, data structures and algorithms, and problem solving. While you're programming, your interviewer may ask you to explain your approach. These aren't trick questions or gotchas. We just want to learn more about your strategy and process. If anything about a question is unclear, you can always ask your interviewer to explain."""

phrases = [
    "The empirical evidence suggests that there is a significant correlation between socioeconomic status and the long-term health outcomes of individuals living in modern urban environments.",
    "The ancient oak tree stood as a silent witness to the passage of time, its twisted branches reaching toward the crimson sky like the weathered fingers of a weary giant.",
    "Democratic institutions require a high level of public trust and active participation from the citizenry to ensure that the rule of law is maintained across all levels of government.",
    "The distributed ledger technology utilizes cryptographic hashes to secure data across a network of decentralized nodes, preventing any single entity from exerting total control over the system.",
    "She felt a profound sense of relief as the plane finally touched down on the runway, knowing that after three long years abroad, she was returning to the comfort of her family.",
    "The Industrial Revolution fundamentally altered the landscape of human civilization by replacing manual labor with mechanized production and shifting populations from rural farms to growing cities.",
    "Rapid urbanization and the expansion of industrial zones have led to the significant degradation of local ecosystems, threatening the biodiversity essential for maintaining environmental balance.",
    "Machine learning algorithms are increasingly being integrated into daily life to automate decision-making processes, ranging from simple product recommendations to complex medical diagnosis.",
    "Cultural identity is a multifaceted construct that evolves through a continuous dialogue between ancient traditions, modern influences, and the personal experiences of individuals.",
    "The human immune system is a sophisticated network of cells and proteins that defends the body against infection by identifying and neutralizing harmful pathogens like bacteria and viruses.",
    "The Amazon rainforest plays a crucial role in global climate regulation by acting as a massive carbon sink and producing a significant portion of the world's oxygen supply.",
    "Effective leadership within a corporate environment involves balancing the immediate need for productivity with the long-term goal of fostering a supportive and innovative team culture.",
    "Astronomers have discovered thousands of exoplanets orbiting distant stars, raising fundamental questions about the probability of life existing elsewhere in the vast expanse of the universe.",
    "Despite the advent of digital communication, the physical library remains a vital community hub where people can access diverse resources and engage in lifelong learning opportunities.", 
    "The complexity of international trade agreements often leads to prolonged negotiations as nations strive to protect their domestic industries while seeking access to new global markets.",
    #"Amidst the ethereal hush of forgotten epochs, the venerable oak, forged by celestial winds and ancestral echoes, remained an unyielding sentinel, observing each fleeting moment with solemn stillness, its gnarled limbs extending into the molten expanse of the twilight sky as if grasping the very essence of temporal decay through the aged, sun-bleached digits of a colossal, melancholic titan long abandoned by nature's ceaseless cycle."
]

scores_original = {}
scores_hardened = {}
changed_phrases = {}
other_metrics = {}

device = "cuda" if torch.cuda.is_available() else "cpu"

def strip_reasoning(text):
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return clean_text

def main():
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    for phrase in phrases:
        # Do the harder phrase generation in a 3 step loop
        for _ in range(3):
            final_output = phrase
            messages = [
                {"role": "system", "content": "You are a helpful assistant that increases the linguistic complexity of text. Please increase the complexity of the given phrase. Only output the rewritten sentence."},
                {"role": "user", "content": f"Original phrase: {final_output}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    repetition_penalty=1.1,
                    temperature=0.4,
                    pad_token_id=tokenizer.pad_token_id
                )

            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            final_output = strip_reasoning(decoded_output)

        scores_hardened[phrase] = sentinel.assign_score([final_output])
        scores_original[phrase] = sentinel.assign_score([phrase])
        changed_phrases[phrase] = final_output

        df = td.extract_metrics(text=phrase, lang="en", metrics=["readability", "information_theory"])
        df_hardened = td.extract_metrics(text=final_output, lang="en", metrics=["readability", "information_theory"])

        other_metrics[phrase] = [df, df_hardened]
        #print(other_metrics[phrase])

def generate_and_score():
    # Ask the model to generate a hard phrase and score it
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    messages = [
        {"role": "system", "content": "You are a helpful assistant that increases the linguistic complexity of text. Please create a hard phrase to understand and understand. Output only the generated phrase."},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            repetition_penalty=1.1,
            temperature=0.4,
            pad_token_id=tokenizer.pad_token_id
        )
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    final_output = strip_reasoning(decoded_output)
    score = sentinel.assign_score([final_output])

    print(f"Generated Phrase: {final_output}")
    print(f"Score: {score}")



if __name__ == "__main__":
    main()


    for phrase in scores_original.keys():
        print("=" * 100)
        print(f"Phrase: {phrase}")
        print(f"Original Score: {scores_original[phrase]}")
        print(f"Hardened Score: {scores_hardened[phrase]}")
        print(f"Changed Phrase: {changed_phrases[phrase]}")
        print(f"Other Metrics: {other_metrics[phrase]}")

    print("=" * 100)

    #sentinel_v_paper = sentinel.assign_score(['Going back up tomorrow; we’re doing stalls and coffin corner practice drills to nail it on the checkride.'])
    #print("Easy test: ", sentinel.assign_score(["The cat is on the table."]))
    #print(f"Rand test: {sentinel_v_paper}")
