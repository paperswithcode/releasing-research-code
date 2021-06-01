import numpy as np
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def GaussianModel(embeddings):
    gmm = GaussianMixture(n_components=1, reg_covar=1e-05)
    gmm.fit(embeddings)

    log_likelihood = gmm.score_samples(embeddings)
    return log_likelihood


def PPCA(embeddings):
    # calculate number of componenets based on 95% variance retention
    n_components = min(embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    var_ratio = pca.explained_variance_ratio_
    y = np.cumsum(var_ratio)
    n_components = int(np.sum(y < 0.95))

    pca = PCA(n_components=n_components)
    pca.fit(embeddings)

    log_likelihood = pca.score_samples(embeddings)
    return log_likelihood


# From https://github.com/clovaai/generative-evaluation-prdc
def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


# From https://github.com/clovaai/generative-evaluation-prdc
def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


# From https://github.com/clovaai/generative-evaluation-prdc
def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii
