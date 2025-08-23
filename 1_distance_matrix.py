from common import load_model_and_tokenizer, extract_attention_from_text
import matplotlib.pyplot as plt
import json
import numpy as np

np.set_printoptions(linewidth=np.inf)

CORRECT_PROMPTS_PATH = "prompts/0_correct_prompts.json"
MISLEADING_PROMPTS_PATH = "prompts/1_misleading_prompts.json"

RANDOM_SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# 1. Load prompts with both correct and misleading hints
with open(CORRECT_PROMPTS_PATH) as f:
    correct_prompts = json.load(f)

with open(MISLEADING_PROMPTS_PATH) as f:
    misleading_prompts = json.load(f)

# 2. Investigate the `attention` and `dist` matrices
tokenizer, model = load_model_and_tokenizer(MODEL_NAME)

correct_prompt = correct_prompts[0]['prompt']

print("Prompt: ", correct_prompt)

attention, _ = extract_attention_from_text(tokenizer, model, correct_prompt)

# Extract the `attention` matrix for the 0th head in the 0th layer
layer = 0
head_idx = 0
A = attention[layer, head_idx].numpy()

# Transform attention matrix A into a distance matrix
print("1) A: \n", A)
print("2) A.T: \n", A.T)
print("3) (A + A.T): \n", (A + A.T))
print("4) (A + A.T) / 2.0: \n", (A + A.T) / 2.0)
print("5) 1.0 - (A + A.T) / 2.0: \n", 1.0 - (A + A.T) / 2.0)