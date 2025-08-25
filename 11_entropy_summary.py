from common import load_model_and_tokenizer, extract_attention_from_text
import matplotlib.pyplot as plt
from tqdm import tqdm
import gudhi as gd
import gudhi.representations
import numpy as np
import json
import random
import os

CORRECT_PROMPTS_PATH = "prompts/correct_prompts.json"
CONFLICTING_PROMPTS_PATH = "prompts/conflicting_prompts.json"
RANDOM_SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-3B"
TEST_SIZE = 500
PREFIX = True

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def barcodes_for_prompt(tokenizer, model, device, prompt, max_dim = 2):
    if PREFIX:
        prompt = 'Is this context correct? Yes/No\n"' + prompt + '"'

    num_layers = model.config.num_hidden_layers  # Get the number of attention layers

    attention = extract_attention_from_text(tokenizer, model, device, prompt)

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

            # # ----- OLD VERSION REMOVING BARCODES WITH INFINITE LENGTH -----
            # _persistence_intervals = np.array([bar for bar in _persistence_intervals if bar[1] != np.inf])

            # ----- BECAUSE MAX DISTANCE IN OUR CASE IS EQUAL TO 1, REPLACE INFINITE ENDS OF BARCODES WITH 1 -----
            for j, bar in enumerate(_persistence_intervals):
                if bar[1] == np.inf:
                    _persistence_intervals[j] = [bar[0], 1.0]

            persistence_intervals_single_layer.append(_persistence_intervals)

        persistence_intervals_multiple_layers.append(persistence_intervals_single_layer)
    return persistence_intervals_multiple_layers


def max_max_death(arr):
    # we create a lambda function which removes the infinity (end = 1 in our case) bars from a barcode.
    remove_infinity = lambda barcode : np.array([bars for bars in barcode if bars[1]!= 1])
    # apply this operator to all barcodes.
    arr = list(map(remove_infinity, arr))

    max_deaths = []
    for single_prompt_persistence_intervals in arr:
        lifespans = np.array([bar[1] for bar in single_prompt_persistence_intervals])
        max_death = np.max(lifespans)
        max_deaths.append(max_death)
    return np.max(max_deaths)


def entropy_summary(persistence_intervals_multiple_prompts, homology_dim, ax, label):
    # H_{h_dim} homology
    # ~h_0_persistence_intervals = persistence_intervals_multiple_prompts[:, :, homology_dim, :]
    h_0_persistence_intervals = []
    for i in range(len(persistence_intervals_multiple_prompts)):
        h_0_persistence_intervals.append([])
        for j in range(len(persistence_intervals_multiple_prompts[i])):
            h_0_persistence_intervals[i].append(persistence_intervals_multiple_prompts[i][j][homology_dim])
    # `h_0_persistence_intervals` has shape PROMPTS_NUM X LAYERS_NUM X BARCODES_NUM

    num_layers = model.config.num_hidden_layers
    ncols = num_layers // 4
    for layer_idx in tqdm(range(model.config.num_hidden_layers)):
    # subplot_idx = 0
    # for layer_idx in [0, num_layers // 3, num_layers // 2, 2 * (num_layers // 3), num_layers - 1]:
        # ~persistence_intervals = h_0_persistence_intervals[:, layer_idx, :]
        _persistence_intervals = []
        for i in range(len(h_0_persistence_intervals)):
            _persistence_intervals.append(h_0_persistence_intervals[i][layer_idx])

        # Get the max distance, for which anything interesting is happening
        max_death = max_max_death(_persistence_intervals)

        resolution = 10000
        ES = gd.representations.Entropy(mode='vector', sample_range=[0, max_death], resolution = resolution, normalized = True)
        es_array = ES.fit_transform(_persistence_intervals)
        es_array = es_array[0]

        xs = np.linspace(0, max_death, len(es_array))

        if label == "Correct prompts":
            ax[layer_idx // ncols, layer_idx % ncols].plot(xs, es_array, "-", color="lightblue", label=label)
            ax[layer_idx // ncols, layer_idx % ncols].set_title(f"Layer: {layer_idx}")
        else:
            ax[layer_idx // ncols, layer_idx % ncols].plot(xs, es_array, "--", color="lightcoral", label=label)
            ax[layer_idx // ncols, layer_idx % ncols].set_title(f"Layer: {layer_idx}")


def entropy_summary_multiple_homologies(tokenizer, model, device, correct_prompts, conflicting_prompts, max_homology_dim, axs):

    # 1. Correct prompts
    persistence_intervals_multiple_prompts = []
    for prompt in tqdm(correct_prompts):
        persistence_intervals_multiple_prompts.append(barcodes_for_prompt(tokenizer, model, device, prompt, max_dim = max_homology_dim + 1))
    # `persistence_intervals_multiple_prompts` has shape PROMPTS_NUM X LAYERS_NUM X HOMOLOGIES_NUM X BARCODES_NUM

    for homology_dim in range(max_homology_dim + 1):
        entropy_summary(persistence_intervals_multiple_prompts, homology_dim, axs[homology_dim], label="Correct prompts")

    # 2. Conflicting prompts
    persistence_intervals_multiple_prompts = []
    for prompt in tqdm(conflicting_prompts):
        persistence_intervals_multiple_prompts.append(barcodes_for_prompt(tokenizer, model, device, prompt, max_dim = max_homology_dim + 1))
    # `persistence_intervals_multiple_prompts` has shape PROMPTS_NUM X LAYERS_NUM X HOMOLOGIES_NUM X BARCODES_NUM

    for homology_dim in range(max_homology_dim + 1):
        entropy_summary(persistence_intervals_multiple_prompts, homology_dim, axs[homology_dim], label="Conflicting prompts")

    # # 3. Add legends to figures
    # for i in range(len(axs)):
    #     axs[i].legend([bps_correct[i]["boxes"][0], bps_conflicting[i]["boxes"][0]], ['Correct prompts', 'Conflicting prompts'])


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

    tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)

    # ----- THE MAIN PART -----
    max_homology_dim = 1

    figs = []
    axs = []
    num_layers = model.config.num_hidden_layers
    for homology_dim in range(max_homology_dim + 1):
        fig, ax = plt.subplots(4, num_layers // 4, figsize = (16, 10))
        # Force scientific notation on Y axes
        for _ax in ax.flat:
            _ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        figs.append(fig)
        axs.append(ax)

    entropy_summary_multiple_homologies(tokenizer, model, device, correct_prompts, conflicting_prompts, max_homology_dim, axs)
    
    for i, fig in enumerate(figs):
        subtitle = MODEL_NAME.split("/")[1]
        if PREFIX:
            fig.subplots_adjust(bottom=0.03, left=0.03, right=0.97, hspace=0.5, wspace=0.5)
            fig.suptitle(rf"{subtitle} | $ES_{{{i}}}$ | prefix: True")
            fig.savefig(f"{subtitle}_ES{i}_n{TEST_SIZE}_pT.png")
            fig.savefig(f"{subtitle}_ES{i}_n{TEST_SIZE}_pT.pdf")
        else:
            fig.subplots_adjust(bottom=0.03, left=0.03, right=0.97, hspace=0.5, wspace=0.5)
            fig.suptitle(rf"{subtitle} | $ES_{{{i}}}$ | prefix: False")
            fig.savefig(f"{subtitle}_ES{i}_n{TEST_SIZE}_pF.png")
            fig.savefig(f"{subtitle}_ES{i}_n{TEST_SIZE}_pF.pdf")

    # plt.show()
