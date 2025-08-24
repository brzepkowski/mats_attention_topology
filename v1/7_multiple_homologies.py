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
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
TEST_SIZE = 3


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
            _persistence_intervals = simplex_tree.persistence_intervals_in_dimension(i)

            # For now just remove barcodes with infinite length. TODO: Fix this later!
            _persistence_intervals = np.array([bar for bar in _persistence_intervals if bar[1] != np.inf])

            persistence_intervals_single_layer.append(_persistence_intervals)

        persistence_intervals_multiple_layers.append(persistence_intervals_single_layer)
    return persistence_intervals_multiple_layers


def custom_entropy(arr):
    entropies = []
    for single_prompt_persistence_intervals in arr:
        k = len(single_prompt_persistence_intervals)
        p = [bar[1] - bar[0] for bar in single_prompt_persistence_intervals]
        p = p/np.sum(p)
        entropy = -np.dot(p, np.log(p))

        # Normalize entropy
        entropy /= np.log(k)
        entropies.append(entropy)
    return np.array(entropies)


def persistence_entropy(persistence_intervals_multiple_prompts, homology_dim, ax):
    # H_{h_dim} homology
    # ~h_0_persistence_intervals = persistence_intervals_multiple_prompts[:, :, homology_dim, :]
    h_0_persistence_intervals = []
    for i in range(len(persistence_intervals_multiple_prompts)):
        h_0_persistence_intervals.append([])
        for j in range(len(persistence_intervals_multiple_prompts[i])):
            h_0_persistence_intervals[i].append(persistence_intervals_multiple_prompts[i][j][homology_dim])
    # `h_0_persistence_intervals` has shape PROMPTS_NUM X LAYERS_NUM X BARCODES_NUM

    pe_per_layer = []
    for layer_idx in tqdm(range(model.config.num_hidden_layers)):
        # ~persistence_intervals = h_0_persistence_intervals[:, layer_idx, :]
        _persistence_intervals = []
        for i in range(len(h_0_persistence_intervals)):
            _persistence_intervals.append(h_0_persistence_intervals[i][layer_idx])

        pe_array = custom_entropy(_persistence_intervals)
        pe_per_layer.append(pe_array)

    bp = ax.boxplot(pe_per_layer, labels=range(model.config.num_hidden_layers), patch_artist=True)
    return bp


def persistence_entropy_multiple_homologies(tokenizer, model, correct_prompts, conflicting_prompts, max_homology_dim, axs):
    bps_correct = []
    bps_conflicting = []

    # 1. Correct prompts
    persistence_intervals_multiple_prompts = []
    for prompt in tqdm(correct_prompts):
        persistence_intervals_multiple_prompts.append(barcodes_for_prompt(tokenizer, model, prompt, max_dim = max_homology_dim + 1))
    # `persistence_intervals_multiple_prompts` has shape PROMPTS_NUM X LAYERS_NUM X HOMOLOGIES_NUM X BARCODES_NUM

    for homology_dim in range(max_homology_dim + 1):
        bp_correct = persistence_entropy(persistence_intervals_multiple_prompts, homology_dim, axs[homology_dim])
        bps_correct.append(bp_correct)
        # Color the boxes differently
        for patch in bp_correct['boxes']:
            patch.set_facecolor('lightblue')

    # 2. Conflicting prompts
    persistence_intervals_multiple_prompts = []
    for prompt in tqdm(conflicting_prompts):
        persistence_intervals_multiple_prompts.append(barcodes_for_prompt(tokenizer, model, prompt, max_dim = max_homology_dim + 1))
    # `persistence_intervals_multiple_prompts` has shape PROMPTS_NUM X LAYERS_NUM X HOMOLOGIES_NUM X BARCODES_NUM

    for homology_dim in range(max_homology_dim + 1):
        bp_conflicting = persistence_entropy(persistence_intervals_multiple_prompts, homology_dim, axs[homology_dim])
        bps_conflicting.append(bp_conflicting)
        # Color the boxes differently
        for patch in bp_conflicting['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.6)  # Make semi-transparent

    # 3. Add legends to figures
    for i in range(len(axs)):
        axs[i].legend([bps_correct[i]["boxes"][0], bps_conflicting[i]["boxes"][0]], ['Correct prompts', 'Conflicting prompts'])


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
    max_homology_dim = 1

    figs = []
    axs = []
    for homology_dim in range(max_homology_dim + 1):
        fig, ax = plt.subplots()
        figs.append(fig)
        axs.append(ax)

    persistence_entropy_multiple_homologies(tokenizer, model, correct_prompts, conflicting_prompts, max_homology_dim, axs)


    # bp_0 = persistence_entropy(tokenizer, model, correct_prompts, ax)
    # bp_1 = persistence_entropy(tokenizer, model, conflicting_prompts, ax)

    # # Color the boxes differently
    # for patch in bp_0['boxes']:
    #     patch.set_facecolor('lightblue')
    # for patch in bp_1['boxes']:
    #     patch.set_facecolor('lightcoral')
    #     patch.set_alpha(0.6)  # Make semi-transparent

    # # Add legend
    # ax.legend([bp_0["boxes"][0], bp_1["boxes"][0]], ['Correct prompts', 'Conflicting prompts'])

    plt.show()
