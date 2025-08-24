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
correct_prompt = correct_prompts[20]

print("correct_prompt: ", correct_prompt)

attention, _ = extract_attention_from_text(tokenizer, model, correct_prompt)

# fig_points = plt.figure(figsize=(16, 8))
fig_barcodes = plt.figure(figsize=(16, 8))

max_dim = 2
max_epsilon = 1.0  # We can now set max_epsilon to 1.0, as we are now operating in the "proper"
                   # space, where the max distance is equal to 1
points=None

_index = 1
chosen_layers = [0, num_layers // 3, num_layers // 2, 2 * (num_layers // 3), num_layers - 1]
for layer_idx in range(num_layers):
    # 3. Get dist matrix as an average over all heads in a given layer
    num_heads = attention[layer_idx].shape[0]
    dist = None
    for head_idx in range(num_heads):
        A = attention[layer_idx, head_idx].numpy()
   
        # MAX-VERSION
        if dist is None:
            dist = np.maximum(A, A.T)
        else:
            dist += np.maximum(A, A.T)
    dist /= num_heads

    print("layer_idx: ", layer_idx)
    # print("dist: \n", dist)
    print("dist.norm: ", np.linalg.norm(dist))  # Just to see how it changes

    if layer_idx in chosen_layers:
        # 4. Embed in 2D space, but use simplex tree from the abstract one, just to have a vague idea of what's happening
        # points, stress = smacof(dist, n_components=2, init=points, n_init=1, random_state=RANDOM_SEED, metric=False)

        rips_complex = gd.RipsComplex(distance_matrix=dist, max_edge_length=max_epsilon)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        simplex_tree.persistence()

        # ax_points = fig_points.add_subplot(4, int(num_layers / 4), layer_idx + 1)  # add_subplot(nrows, ncols, index, **kwargs)
        # ax_barcodes = fig_barcodes.add_subplot(4, int(num_layers / 4), layer_idx + 1)  # add_subplot(nrows, ncols, index, **kwargs)

        # ax_points = fig_points.add_subplot(1, len(chosen_layers), _index)  # add_subplot(nrows, ncols, index, **kwargs)
        ax_barcodes = fig_barcodes.add_subplot(1, len(chosen_layers), _index)  # add_subplot(nrows, ncols, index, **kwargs)

        # draw_simplicial_complex(ax_points, points, simplex_tree, layer_idx)
        gd.plot_persistence_barcode(simplex_tree.persistence_intervals_in_dimension(1), axes=ax_barcodes, fontsize=10)

        _index += 1
plt.show()
