from common import load_model_and_tokenizer, extract_attention_from_text
from sklearn.manifold import smacof
import matplotlib.pyplot as plt
import gudhi as gd
import gudhi.representations
import numpy as np
import json
import random
from tqdm import tqdm

CORRECT_PROMPTS_PATH = "prompts/correct_prompts.json"
CONFLICTING_PROMPTS_PATH = "prompts/conflicting_prompts.json"
RANDOM_SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-3B"
TEST_SIZE = 10


def barcodes_for_prompt(tokenizer, model, prompt, max_dim = 2):
    num_layers = model.config.num_hidden_layers  # Get the number of attention layers

    attention = extract_attention_from_text(tokenizer, model, prompt)

    max_epsilon = 1.0  # We can now set max_epsilon to 1.0, as we are now operating in the "proper"
                       # space, where the max distance is equal to 1

    persistence_intervals_multiple_layers = []
    for layer_idx in range(num_layers):
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

        rips_complex = gd.RipsComplex(distance_matrix=dist, max_edge_length=max_epsilon)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        simplex_tree.persistence()

        persistence_intervals_single_layer = []
        for i in range(max_dim):
            persistence_intervals_single_layer.append(simplex_tree.persistence_intervals_in_dimension(i))

        persistence_intervals_multiple_layers.append(persistence_intervals_single_layer)
    return persistence_intervals_multiple_layers


def persistence_entropy(tokenizer, model, prompts, ax):
    persistence_intervals_multiple_prompts = []
    for prompt in tqdm(prompts):
        persistence_intervals_multiple_prompts.append(barcodes_for_prompt(tokenizer, model, prompt, max_dim = 2))
    # `persistence_intervals_multiple_prompts` has shape PROMPTS_NUM X LAYERS_NUM X HOMOLOGIES_NUM X BARCODES_NUM

    # print("0) len(persistence_intervals_multiple_prompts): ", len(persistence_intervals_multiple_prompts))
    # print("1) len(persistence_intervals_multiple_prompts[0]): ", len(persistence_intervals_multiple_prompts[0]))
    # print("2) len(persistence_intervals_multiple_prompts[0][0]): ", len(persistence_intervals_multiple_prompts[0][0]))
    # print("3) len(persistence_intervals_multiple_prompts[0][0][0]): ", len(persistence_intervals_multiple_prompts[0][0][0]))
    # print("=" * 100)

    # H_0 homology
    # ~h_0_persistence_intervals = persistence_intervals_multiple_prompts[:, :, 0, :]
    h_0_persistence_intervals = []
    for i in range(len(persistence_intervals_multiple_prompts)):
        h_0_persistence_intervals.append([])
        for j in range(len(persistence_intervals_multiple_prompts[i])):
            h_0_persistence_intervals[i].append(persistence_intervals_multiple_prompts[i][j][0])
    # `h_0_persistence_intervals` has shape PROMPTS_NUM X LAYERS_NUM X BARCODES_NUM

    # print("0) len(h_0_persistence_intervals): ", len(h_0_persistence_intervals))
    # print("1) len(h_0_persistence_intervals[0]): ", len(h_0_persistence_intervals[0]))
    # print("2) len(h_0_persistence_intervals[0][0]): ", len(h_0_persistence_intervals[0][0]))
    # print("=" * 100)

    pe_per_layer = []
    for layer_idx in tqdm(range(model.config.num_hidden_layers)):
        # ~persistence_intervals = h_0_persistence_intervals[:, layer_idx, :]
        _persistence_intervals = []
        for i in range(len(h_0_persistence_intervals)):
            _persistence_intervals.append(h_0_persistence_intervals[i][layer_idx])

        # print("0) len(_persistence_intervals): ", len(_persistence_intervals))
        # print("1) len(_persistence_intervals[0]): ", len(_persistence_intervals[0]))
        # print("-" * 100)

        # For now just remove barcodes with infinite length. TODO: Fix this later!
        remove_infinity = lambda barcode : np.array([bars for bars in barcode if bars[1]!= np.inf])
        # apply this operator to all barcodes.
        _persistence_intervals = list(map(remove_infinity, _persistence_intervals))

        # print("0) len(_persistence_intervals): ", len(_persistence_intervals))
        # print("1) len(_persistence_intervals[0]): ", len(_persistence_intervals[0]))

        PE = gd.representations.Entropy()
        pe = PE.fit_transform(_persistence_intervals)
        pe_array = np.array(pe[:,0])

        pe_per_layer.append(pe_array)

    bp = ax.boxplot(pe_per_layer, labels=range(model.config.num_hidden_layers), patch_artist=True)
    return bp


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    random.seed(RANDOM_SEED)

    with open(CORRECT_PROMPTS_PATH) as f:
        correct_prompts = json.load(f)

    with open(CONFLICTING_PROMPTS_PATH) as f:
        conflicting_prompts = json.load(f)

    prompts_indices = sorted(random.choices(range(len(correct_prompts)), k=TEST_SIZE))
    print("prompts_indices: ", prompts_indices)

    correct_prompts = np.array(correct_prompts)[prompts_indices]
    conflicting_prompts = np.array(conflicting_prompts)[prompts_indices]

    tokenizer, model = load_model_and_tokenizer(MODEL_NAME)

    # ----- THE MAIN PART -----
    fig, ax = plt.subplots()
    bp_0 = persistence_entropy(tokenizer, model, correct_prompts, ax)
    bp_1 = persistence_entropy(tokenizer, model, conflicting_prompts, ax)

    # Color the boxes differently
    for patch in bp_0['boxes']:
        patch.set_facecolor('lightblue')
    for patch in bp_1['boxes']:
        patch.set_facecolor('lightcoral')

    # Add legend
    ax.legend([bp_0["boxes"][0], bp_1["boxes"][0]], ['Correct prompts', 'Conflicting prompts'])

    plt.show()
