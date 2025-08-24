from common import prompt_len_with_most_occurences, keep_prompts_of_len


if __name__ == "__main__":
    # # 1. Correct prompts
    # print("=" * 20, "CORRECT PROMPTS", "=" * 20)
    # prompts_path = "prompts/0_correct_prompts.json"
    
    # output_path = "prompts/qwen_0_5B_correct_prompts.json"
    # model_name = "Qwen/Qwen2.5-0.5B"
    # target_length = prompt_len_with_most_occurences(model_name, prompts_path)
    # keep_prompts_of_len(target_length, model_name, prompts_path, output_path)

    # output_path = "prompts/qwen_1_5B_correct_prompts.json"
    # model_name = "Qwen/Qwen2.5-1.5B"
    # target_length = prompt_len_with_most_occurences(model_name, prompts_path)
    # keep_prompts_of_len(target_length, model_name, prompts_path, output_path)

    # output_path = "prompts/deepseek_1_3B_correct_prompts.json"
    # model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    # target_length = prompt_len_with_most_occurences(model_name, prompts_path)
    # keep_prompts_of_len(target_length, model_name, prompts_path, output_path)
    
    # 2. Wrong prompts
    print("=" * 20, "WRONG PROMPTS", "=" * 20)
    prompts_path = "prompts/1_wrong_prompts.json"

    output_path = "prompts/qwen_0_5B_wrong_prompts.json"
    model_name = "Qwen/Qwen2.5-0.5B"
    target_length = prompt_len_with_most_occurences(model_name, prompts_path)
    keep_prompts_of_len(target_length, model_name, prompts_path, output_path)

    output_path = "prompts/qwen_1_5B_wrong_prompts.json"
    model_name = "Qwen/Qwen2.5-1.5B"
    target_length = prompt_len_with_most_occurences(model_name, prompts_path)
    keep_prompts_of_len(target_length, model_name, prompts_path, output_path)

    output_path = "prompts/deepseek_1_3B_wrong_prompts.json"
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    target_length = prompt_len_with_most_occurences(model_name, prompts_path)
    keep_prompts_of_len(target_length, model_name, prompts_path, output_path)