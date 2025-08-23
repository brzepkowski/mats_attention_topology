from common import load_model_and_tokenizer, extract_attention_from_text, draw_simplicial_complex
from sklearn.manifold import smacof
import matplotlib.pyplot as plt
import gudhi as gd
import numpy as np
import json

np.set_printoptions(linewidth=np.inf)

CORRECT_PROMPTS_PATH = "prompts/0_correct_prompts.json"
MISLEADING_PROMPTS_PATH = "prompts/1_misleading_prompts.json"

RANDOM_SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# 1. Load prompts with both correct and misleading hints
with open(CORRECT_PROMPTS_PATH) as f:
    correct_prompts = json.load(f)

# with open(MISLEADING_PROMPTS_PATH) as f:
#     misleading_prompts = json.load(f)

# 2. Investigate the `attention` and `dist` matrices
tokenizer, model = load_model_and_tokenizer(MODEL_NAME)
num_layers = model.config.num_hidden_layers  # Get the number of attention layers
correct_prompt = correct_prompts[0]['prompt']
attention, _ = extract_attention_from_text(tokenizer, model, correct_prompt)

# Extract the `attention` matrix for the 0th head in each layer
head_idx = 0
points = None  # We want to use points from the previous layer as a starting point, while generating the embedding for the next layer
fig = plt.figure(figsize=(16, 8))

epsilon = 0.5  # For now fix the epsilon to 0.5

for layer_idx in range(num_layers):
    A = attention[layer_idx, head_idx].numpy()

    dist = 1.0 - (A + A.T) / 2.0

    # 2. Embed tokens in 2D space
    # Note: Because dist cannot be interpreted as a proper metric (rather as dissimilarity),
    # this embedding will come with some error! (The last attribute below is thus crucial!)
    points, stress = smacof(dist, n_components=2, init=points, n_init=1, random_state=RANDOM_SEED, metric=False)

    # 3. Plot the simplical complexes for different values of epsilon
    max_dim = 2
    ax = fig.add_subplot(4, int(num_layers / 4), layer_idx + 1)  # add_subplot(nrows, ncols, index, **kwargs)

    # Create Vietoris–Rips complex - a simplex is included iff all its vertices are pairwise within distance epsilon
    rips_complex = gd.RipsComplex(points=points, max_edge_length=epsilon)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        
    draw_simplicial_complex(ax, points, simplex_tree, layer_idx)

plt.tight_layout()
plt.show()
