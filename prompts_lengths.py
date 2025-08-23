from transformers import AutoTokenizer
import numpy as np
import json


def get_number_of_tokens(prompt_text, tokenizer):
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True)
    return inputs["input_ids"].shape[1]

def prompt_len_with_most_occurences(model_name, prompts_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    prompts_list = []
    with open(prompts_path) as f:
        prompts_list = json.load(f)

    # Count number of tokens with given length
    prompts_lenghts = {}
    for entry in prompts_list:
        prompt_len = get_number_of_tokens(entry["prompt"], tokenizer)
        # print("entry: ", entry, " | prompt_len: ", prompt_len)
        if prompt_len not in prompts_lenghts:
            prompts_lenghts[prompt_len] = 1
        else:
            prompts_lenghts[prompt_len] += 1
    print("prompts_lenghts: ", prompts_lenghts)

    # Get the most often appearing number of tokens
    final_prompt_len = -1
    most_occurences = np.max(list(prompts_lenghts.values()))
    for prompt_len, occurences in prompts_lenghts.items():
        if occurences == most_occurences:
            final_prompt_len = prompt_len
            break
    print("Most occurences: ", most_occurences)
    print("Target prompt length: ", final_prompt_len)

    return final_prompt_len


def keep_prompts_of_len(target_length, model_name, prompts_path, output_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    prompts_list = []
    with open(prompts_path) as f:
        prompts_list = json.load(f)

    kept_prompts = []
    for entry in prompts_list:
        prompt_len = get_number_of_tokens(entry["prompt"], tokenizer)
        if prompt_len == target_length:
            kept_prompts.append(entry)

    # print("kept_prompts: \n", kept_prompts)

    with open(output_path, 'w') as f:
        json.dump(kept_prompts, f)



if __name__ == "__main__":
    prompts_path = "prompts/initial_prompts.json"
    output_path = "prompts/qwen_0_5B_prompts.json"
    model_name = "Qwen/Qwen2.5-0.5B"

    target_length = prompt_len_with_most_occurences(model_name, prompts_path)
    keep_prompts_of_len(target_length, model_name, prompts_path, output_path)

    # prompt_len_with_most_occurences("Qwen/Qwen2.5-1.5B", prompts_path)
    # prompt_len_with_most_occurences("deepseek-ai/deepseek-coder-1.3b-base", prompts_path)
    