import torch
import numpy as np
from tqdm import tqdm


class CoresetSampler:
    """
    Coreset Sampler for selecting representative samples from embeddings.

    Attributes:
        n_samples (int): Number of samples to select.
        initialization (str): Initialization method for the sampling.
        device (str): Device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        dbscan_params (dict): Parameters for DBScan clustering if using "dbscan" initialization.
        tqdm_disable (bool): Whether to disable tqdm progress bar.
        verbose (int): Verbosity level.

    Methods:
        __init__(self, n_samples=100, initialization="", device="cuda", dbscan_params={}, tqdm_disable=True, verbose=0):
            Constructor.

        initialize(self, embeddings, force=True):
            Initialize the sampler.

        sample(self, embeddings, init_ids=None, n_samples=None):
            Sample representative points from embeddings.
    """

    def __init__(
        self,
        n_samples=100,
        initialization="",
        device="cuda",
        dbscan_params={},
        tqdm_disable=True,
        verbose=0,
    ):
        """
        Constructor.

        Args:
            n_samples (int): Number of samples to select.
            initialization (str): Initialization method for the sampling. Set the value to "dbscan" for DBScan init.
            device (str): Device to use, e.g., "cuda" for GPU or "cpu" for CPU.
            dbscan_params (dict): Parameters for DBScan clustering if using "dbscan" initialization.
            tqdm_disable (bool): Whether to disable tqdm progress bar.
            verbose (int): Verbosity level.
        """
        self.n_samples = n_samples
        self.initialization = initialization
        self.device = device
        self.dbscan_params = dbscan_params
        self.tqdm_disable = tqdm_disable
        self.verbose = verbose

        self.dbscan_y = None

    def initialize(self, embeddings, force=True):
        """
        Initialize the sampler.

        If using "dbscan" initialization, the method will use the centroids of the clusters found by DBScan.
        Note that this can be slow with big datasets, especially on GPU.
        Otherwise, it uses a random data point.

        Args:
            embeddings (numpy.ndarray): Input embeddings.
            force (bool): Force re-computation of initialization.

        Returns:
            list: List of initialized sample indices.
        """

        init_ids = [np.random.choice(len(embeddings))]

        if self.initialization == "dbscan":
            if self.dbscan_y is not None and not force:
                if self.verbose:
                    print("Use previously computed dbscan_y")
                y = self.dbscan_y

            else:
                try:
                    from cuml import DBSCAN
                    if self.verbose:
                        print("Using cuml gpu-accelerated algorithms")
                except ImportError:
                    from sklearn.cluster import DBSCAN

                    if self.verbose:
                        print("Using sklearn algorithms")

                if self.verbose:
                    print("Initializing with DBScan")
                dbscan = DBSCAN(**self.dbscan_params)
                dbscan = dbscan.fit(embeddings)

                try:  # cuml
                    y = dbscan.labels_.values.get()
                except AttributeError:
                    y = dbscan.labels_
                self.dbscan_y = y

            counts = np.bincount(y + 1)
            biggest = np.argsort(counts[1:])[::-1]
            init_ids = [np.random.choice(np.where(y == l)[0]) for l in biggest]

        if self.verbose:
            print(f"Initialize with {len(init_ids)} points")

        return init_ids

    def sample(self, embeddings, init_ids=None, n_samples=None):
        """
        Sample representative points from embeddings.

        Args:
            embeddings (numpy.ndarray): Input embeddings.
            init_ids (list, optional): List of initialized sample indices. Defaults to None.
                If None, calls the self.initialize method.
            n_samples (int, optional): Number of samples to select. Defaults to None.
                If None, use value specified during initialization.

        Returns:
            list: List of sampled indices.
        """
        if init_ids is None:
            ids = self.initialize(embeddings, force=False)
        else:
            ids = np.copy(init_ids).tolist()

        if n_samples is None:
            n_samples = self.n_samples

        # To tensor
        embeddings = torch.from_numpy(embeddings).to(self.device)

        # Compute minimum distances to the initialization ids of all embeddings
        if len(ids) > 1:
            min_distances = [
                ((embeddings[id_].unsqueeze(0) - embeddings) ** 2).sum(
                    -1, keepdims=True
                )
                for id_ in ids[:-1]
            ]
            min_distances = torch.cat(min_distances, 1).amin(1, keepdims=True)
        else:
            min_distances = torch.empty(len(embeddings), 0).to(self.device)

        for i in tqdm(range(len(ids), n_samples), disable=self.tqdm_disable):
            current = embeddings[ids[-1]]  # Last appended id

            # Compute distances to the last sampled point
            new_dist = ((current.unsqueeze(0) - embeddings) ** 2).sum(-1, keepdims=True)

            # Update the minimum distance using(min(all_previous, current))
            min_distances = torch.cat([min_distances, new_dist], 1).amin(
                1, keepdims=True
            )

            # Sample the farthest point
            fartherst = min_distances.argmax().item()
            ids.append(fartherst)

        return ids
