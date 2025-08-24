from common import load_model_and_tokenizer, answer_prompt
import json
import torch
import random
import numpy as np
MODEL_NAME = "Qwen/Qwen2.5-3B"


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    conflicting_prompts_path = "prompts/all_conflicting_prompts.json"

    tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)

    with open(conflicting_prompts_path) as f:
        conflicting_prompts_list = json.load(f)

    # 1. Answer the bare question, without any hint
    kept_indices = []
    for i, prompt in enumerate(conflicting_prompts_list):
        print("=" * 50)
        # prompt = 'Is there an error in the following paragraph? Answer with one word: "YES" or "NO".\n"' + prompt + '"'

        # 1. Check if there is an error in the paragraph
        # prompt = 'Does the given context conflict with what you know? Yes/No \n"' + prompt + '"'
        # prompt = 'How strongly does the given context conflict with what you know on a scale from 1 to 10?\n"' + prompt + '"'
        prompt = 'Is this context correct? Yes/No\n"' + prompt + '"'
        model_answer = answer_prompt(tokenizer, model, device, prompt)

        print(f"===== {i} ======\nANSWER:{model_answer}")
        if not "yes" in model_answer.lower():
            kept_indices.append(i)

    correct_prompts_path = "prompts/all_correct_prompts.json"
    conflicting_prompts_path = "prompts/all_conflicting_prompts.json"

    with open(conflicting_prompts_path) as f:
        conflicting_prompts_list = np.array(json.load(f))

    with open(correct_prompts_path) as f:
        correct_prompts_list = np.array(json.load(f))

    conflicting_prompts_list = list(conflicting_prompts_list[kept_indices])
    correct_prompts_list = list(correct_prompts_list[kept_indices])

    with open("prompts/conflicting_prompts.json", "w") as f:
        json.dump(conflicting_prompts_list, f)

    with open("prompts/correct_prompts.json", "w") as f:
        json.dump(correct_prompts_list, f)

