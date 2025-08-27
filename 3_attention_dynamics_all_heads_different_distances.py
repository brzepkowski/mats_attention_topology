from common import load_model_and_tokenizer, extract_attention_from_text, draw_simplicial_complex
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.cluster import KMeans
from sklearn.manifold import smacof
import matplotlib.pyplot as plt
import gudhi as gd
import numpy as np
import json

np.set_printoptions(linewidth=np.inf)

CORRECT_PROMPTS_PATH = "prompts/correct_prompts.json"
# MISLEADING_PROMPTS_PATH = "prompts/1_misleading_prompts.json"

RANDOM_SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-3B"



def plot_clusters_in_insets(points, ax):
    """
    Clusters 2D points into two clusters and plots them in separate insets.

    Parameters
    ----------
    points : array-like, shape (n_samples, 2)
        The 2D points to cluster.
    ax : matplotlib.axes.Axes
        The main axis to draw insets on.
    """
    points = np.asarray(points)

    # Run KMeans clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(points)

    # Create inset axes for each cluster
    insets = []
    positions = [(0.3, 0.04, "50%", "50%"),  # bottom-right
                 (0.05, 0.57, "30%", "30%")]  # top-left

    for cluster_id, pos in zip(range(2), positions):
        # Create inset axis
        ax_inset = inset_axes(ax, width=pos[2], height=pos[3],
                              loc=3, bbox_to_anchor=(pos[0], pos[1], 1, 1),
                              bbox_transform=ax.transAxes, borderpad=1)

        # Extract cluster points
        cluster_points = points[labels == cluster_id]

        # Scatter plot
        ax_inset.scatter(cluster_points[:, 0], cluster_points[:, 1], s=40)
        # if cluster_id == 0:
        #     print("min(x): ", np.min(cluster_points[:, 0]), "\t | max(x): ", np.max(cluster_points[:, 0]), "\t | min(y): ", np.min(cluster_points[:, 1]), "\t | ", "max(y): ", np.max(cluster_points[:, 1]))
            # ax_inset.set_xlim(-0.05, 0.11)
            # ax_inset.set_ylim(-0.44, -0.26)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        # ax_inset.set_title(f"Cluster {cluster_id+1}", fontsize=8)
        ax_inset.tick_params(labelsize=6)

        insets.append(ax_inset)

    # Optionally show cluster centers on main ax
    ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="coolwarm", alpha=0.3)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c="black", marker="x", s=100, label="Centers")
    # ax.legend()

    return insets


def main():
    tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)
    num_layers = model.config.num_hidden_layers  # Get the number of attention layers

    attention = extract_attention_from_text(tokenizer, model, device, "Hello world!")

    fig_dist = plt.figure(figsize=(8, 4))

    max_dim = 2
    max_epsilon = 0.7

    points = None

    # for layer_idx in range(num_layers):
    subplot_idx = 1
    for layer_idx in [0, 12, 24, 35]:

        # 3. Get dist matrix as an average over all heads in a given layer
        num_heads = attention[layer_idx].shape[0]

        dist = None
        for head_idx in range(num_heads):
            
            A = attention[layer_idx, head_idx].numpy()
            if dist is None:
                dist = np.maximum(A, A.T)
                if layer_idx == 0 and head_idx == 0:
                    print("A: \n", A)
                    print("dist: \n", dist)
            else:
                dist += np.maximum(A, A.T)
        dist /= num_heads

        if layer_idx in [0, 2]:
            print("dist: \n", dist)

        # 4. Embed tokens in 2D space
        # Note: Because dist cannot be interpreted as a proper metric (rather as dissimilarity),
        # this embedding will come with some error! (The last attribute below is thus crucial!)
        points, stress = smacof(dist, n_components=2, init=points, n_init=1, random_state=RANDOM_SEED, metric=False)
        # print("stress: ", stress)

        # 5. Plot the simplical complexes for different values of epsilon
        # ax = fig_dist.add_subplot(4, int(num_layers / 4), layer_idx + 1)  # add_subplot(nrows, ncols, index, **kwargs)
        ax = fig_dist.add_subplot(1, 4, subplot_idx)  # add_subplot(nrows, ncols, index, **kwargs)

        # Create Vietoris–Rips complex - a simplex is included iff all its vertices are pairwise within distance epsilon
        rips_complex = gd.RipsComplex(points=points, max_edge_length=max_epsilon)

        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        
        if layer_idx < 2:
            draw_simplicial_complex(ax, points, simplex_tree, layer_idx)
        else:
            plot_clusters_in_insets(points, ax)
            ax.set_title(f"LAYER: {layer_idx}", fontsize=12)
        subplot_idx += 1

    fig_dist.suptitle(r"$dist = max(A, A.T)$")
    plt.tight_layout()

    fig_dist.savefig("dist.pdf")
    fig_dist.savefig("dist.png")
    # plt.show()


if __name__ == "__main__":
    main()
