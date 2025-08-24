from common import load_model_and_tokenizer, extract_attention_from_text, draw_simplicial_complex
from sklearn.manifold import smacof
import matplotlib.pyplot as plt
import gudhi as gd
import numpy as np
import json

np.set_printoptions(linewidth=np.inf)

CORRECT_PROMPTS_PATH = "prompts/correct_prompts.json"
# MISLEADING_PROMPTS_PATH = "prompts/1_misleading_prompts.json"

RANDOM_SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# 1. Load prompts with both correct and misleading hints
with open(CORRECT_PROMPTS_PATH) as f:
    correct_prompts = json.load(f)

# 2. Investigate the `attention` and `dist` matrices
tokenizer, model = load_model_and_tokenizer(MODEL_NAME)
num_layers = model.config.num_hidden_layers  # Get the number of attention layers

attention = extract_attention_from_text(tokenizer, model, "This is just some test prompt!")

fig_dist_0 = plt.figure(figsize=(16, 8))
fig_dist_1 = plt.figure(figsize=(16, 8))

max_dim = 2
max_epsilon = 0.7

points_0 = None
points_1 = None

for layer_idx in range(num_layers):

    # 3. Get dist matrix as an average over all heads in a given layer
    num_heads = attention[layer_idx].shape[0]
    dist_0 = None
    for head_idx in range(num_heads):
        A = attention[layer_idx, head_idx].numpy()
        if dist_0 is None:
            dist_0 = 1.0 - np.maximum(A, A.T)
        else:
            dist_0 += 1.0 - np.maximum(A, A.T)
    dist_0 /= num_heads

    dist_1 = None
    for head_idx in range(num_heads):
        A = attention[layer_idx, head_idx].numpy()
        if dist_1 is None:
            dist_1 = np.maximum(A, A.T)
        else:
            dist_1 += np.maximum(A, A.T)
    dist_1 /= num_heads

    if layer_idx in [0, 10, 27]:
        print("layer: ", layer_idx)

        # print("dist_0: \n", dist_0)
        # print("norm_0: ", np.linalg.norm(dist_0))
        
        print("dist_1: \n", dist_1)
        print("norm_1: ", np.linalg.norm(dist_1))

    # 4. Embed tokens in 2D space
    # Note: Because dist cannot be interpreted as a proper metric (rather as dissimilarity),
    # this embedding will come with some error! (The last attribute below is thus crucial!)
    points_0, stress_0 = smacof(dist_0, n_components=2, init=points_0, n_init=1, random_state=RANDOM_SEED, metric=False)
    # if layer_idx in [0, 10, 27]:
    #     print("stress_0: ", stress_0)
    points_1, stress_1 = smacof(dist_1, n_components=2, init=points_1, n_init=1, random_state=RANDOM_SEED, metric=False)
    if layer_idx in [0, 10, 27]:
        print("stress_1: ", stress_1)

    # 5. Plot the simplical complexes for different values of epsilon
    ax_0 = fig_dist_0.add_subplot(4, int(num_layers / 4), layer_idx + 1)  # add_subplot(nrows, ncols, index, **kwargs)
    ax_1 = fig_dist_1.add_subplot(4, int(num_layers / 4), layer_idx + 1)  # add_subplot(nrows, ncols, index, **kwargs)

    # Create Vietoris–Rips complex - a simplex is included iff all its vertices are pairwise within distance epsilon
    rips_complex_0 = gd.RipsComplex(points=points_0, max_edge_length=max_epsilon)
    rips_complex_1 = gd.RipsComplex(points=points_1, max_edge_length=max_epsilon)

    simplex_tree_0 = rips_complex_0.create_simplex_tree(max_dimension=max_dim)
    simplex_tree_1 = rips_complex_1.create_simplex_tree(max_dimension=max_dim)
        
    draw_simplicial_complex(ax_0, points_0, simplex_tree_0, layer_idx)
    draw_simplicial_complex(ax_1, points_1, simplex_tree_1, layer_idx)


fig_dist_0.suptitle(r"$dist = 1 - max(A, A.T)$")
fig_dist_1.suptitle(r"$dist = max(A, A.T)$")
plt.tight_layout()

fig_dist_0.savefig("dist_0.pdf")
fig_dist_1.savefig("dist_1.pdf")
# plt.show()
