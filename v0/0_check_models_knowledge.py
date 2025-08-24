from common import load_model_and_tokenizer, answer_prompt
import json

if __name__ == "__main__":
    prompts_path = "prompts/0_correct_prompts.json"

    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer, model = load_model_and_tokenizer(model_name)

    with open(prompts_path) as f:
        prompts_list = json.load(f)

    # 1. Answer the bare question, without any hint
    for entry in prompts_list:
        print("=" * 50)
        prompt = entry["prompt"].split(".")[1]
        prompt = "Answer the question with one word. " + prompt
        model_answer = answer_prompt(prompt, tokenizer, model)

        print("Q:", prompt, ", A: ", entry["answer"], ", MA:", model_answer)

    # 2. Answer the question with a hint
    for entry in prompts_list:
        print("=" * 50)
        prompt = "Answer the question with one word. " + entry["prompt"]
        model_answer = answer_prompt(prompt, tokenizer, model)

        print("Q:", prompt, ", A: ", entry["answer"], ", MA:", model_answer)

    # 3. Answer the question with a misleading hint
    misleading_prompts_path = "prompts/1_misleading_prompts.json"

    with open(misleading_prompts_path) as f:
        misleading_prompts_list = json.load(f)

    for entry in misleading_prompts_list:
        print("=" * 50)
        prompt = "Answer the question with one word. " + entry["prompt"]
        model_answer = answer_prompt(prompt, tokenizer, model)

        print("Q:", prompt, ", A: ", entry["answer"], ", MA:", model_answer)

