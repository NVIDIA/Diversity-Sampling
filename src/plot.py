import numpy as np
import matplotlib.pyplot as plt

try:
    from cuml import TSNE
except ImportError:
    from sklearn.manifold import TSNE


def plot_tsne(x, y, counts, min_size=100, figsize=(20, 20), legend=True, title=""):
    """
    Plot a t-SNE visualization of data points.

    Args:
        x (numpy.ndarray): 2D array containing t-SNE coordinates.
        y (numpy.ndarray): Array of cluster labels.
        counts (numpy.ndarray): Array of cluster counts.
        min_size (int, optional): Minimum size threshold for highlighting clusters. Defaults to 100.
        figsize (tuple, optional): Figure size. Defaults to (20, 20).
        legend (bool, optional): Whether to show the legend. Defaults to True.
        title (str, optional): Title for the plot. Defaults to "".
    """
    for label in np.unique(y):
        kept = np.where(y == label)[0]

        if label == -1:
            plt.scatter(x[kept, 0], x[kept, 1], s=0.1, c="gray")
        else:
            if counts[label + 1] > min_size:
                plt.scatter(x[kept, 0], x[kept, 1], s=10, label=f"Cluster {label}")
            else:
                plt.scatter(x[kept, 0], x[kept, 1], s=1)

    if legend:
        lgd = plt.legend(fontsize=10)
        for i in range(len(lgd.legend_handles)):
            lgd.legend_handles[i]._sizes = [30]

    plt.axis(False)

    if title:
        plt.title(title, fontsize=15)


def plot_dbscan_results(embeds, y, counts, min_size=100):
    """
    Plot the results of DBScan clustering using t-SNE visualization.
    Note that data has to be subsampled to ~10000 points for t-SNE to converge.

    Args:
        embeds (numpy.ndarray): 2D array containing embeddings.
        y (numpy.ndarray): Array of cluster labels.
        counts (numpy.ndarray): Array of cluster counts.
        min_size (int, optional): Minimum size threshold for highlighting clusters. Defaults to 100.
    """
    S = len(embeds) // 10000

    tsne = TSNE(
        n_components=2,
        perplexity=10,
        early_exaggeration=10,
    )

    X_embedded = tsne.fit_transform(embeds[::S])

    try:
        X_embedded = X_embedded.values.get()
    except AttributeError:
        pass

    plt.figure(figsize=(20, 20))
    plot_tsne(X_embedded, y[::S], counts, min_size=min_size)
    plt.show()


def plot_coreset_results(embeds, y, counts, coreset_ids, min_size=100):
    """
    Plot the results before and after applying a Coreset sampling using t-SNE visualization.

    Args:
        embeds (numpy.ndarray): 2D array containing embeddings.
        y (numpy.ndarray): Array of cluster labels.
        counts (numpy.ndarray): Array of cluster counts.
        coreset_ids (list): List of coreset sample indices.
        min_size (int, optional): Minimum size threshold for highlighting clusters. Defaults to 100.
    """
    S = len(embeds) // 10000

    # Subsample
    kept_tsne = np.concatenate([np.arange(len(embeds))[::S], coreset_ids])
    kept_tsne = np.sort(np.unique(kept_tsne))
    kept_coreset = np.array(
        [idx for idx, v in enumerate(kept_tsne) if v in coreset_ids]
    )

    # T-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=50,
        early_exaggeration=200,
    )

    X_embedded = tsne.fit_transform(embeds[kept_tsne])

    try:
        X_embedded = X_embedded.values.get()
    except AttributeError:
        pass

    # Plot results
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plot_tsne(
        X_embedded,
        y[kept_tsne],
        counts,
        min_size=min_size,
        legend=False,
        title="Before Coreset",
    )

    plt.subplot(1, 2, 2)
    plot_tsne(
        X_embedded[kept_coreset],
        y[kept_tsne][kept_coreset],
        counts,
        min_size=min_size,
        legend=False,
        title="After Coreset",
    )

    plt.show()
