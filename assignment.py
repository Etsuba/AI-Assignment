import numpy as np


def compute_distance_matrix(data):
    """
    Compute pairwise Euclidean distance matrix.
    """
    sq_norms = np.sum(data ** 2, axis=1).reshape(-1, 1)
    dist_sq = sq_norms + sq_norms.T - 2 * data @ data.T
    np.maximum(dist_sq, 0, out=dist_sq)
    return np.sqrt(dist_sq)


def ward_distance(cluster_a, cluster_b, data):
    """
    Compute Ward linkage distance between two clusters.
    """
    A = data[cluster_a]
    B = data[cluster_b]

    mean_A = np.mean(A, axis=0)
    mean_B = np.mean(B, axis=0)
    mean_AB = np.mean(np.vstack([A, B]), axis=0)

    n_A = len(A)
    n_B = len(B)

    # Increase in within-cluster sum of squares
    return (
        n_A * np.sum((mean_A - mean_AB) ** 2)
        + n_B * np.sum((mean_B - mean_AB) ** 2)
    )


def linkage_distance(cluster_a, cluster_b, dist_matrix, data, linkage):
    """
    Compute distance between two clusters using linkage criteria.
    """
    if linkage == "ward":
        return ward_distance(cluster_a, cluster_b, data)

    distances = dist_matrix[np.ix_(cluster_a, cluster_b)]

    if linkage == "single":
        return np.min(distances)
    elif linkage == "complete":
        return np.max(distances)
    elif linkage == "average":
        return np.mean(distances)
    else:
        raise ValueError(f"Unsupported linkage: {linkage}")


def agglomerative_hierarchical_clustering(data, linkage="average"):
    """
    Full agglomerative hierarchical clustering algorithm.

    Parameters:
        data (np.ndarray): Shape (n_samples, n_features)
        linkage (str): 'single', 'complete', 'average', or 'ward'

    Returns:
        merges (np.ndarray): Merge history for dendrogram
    """
    n_samples = data.shape[0]
    dist_matrix = compute_distance_matrix(data)

    clusters = {i: [i] for i in range(n_samples)}
    next_cluster_id = n_samples
    merges = []

    while len(clusters) > 1:
        min_dist = np.inf
        pair_to_merge = None

        cluster_ids = list(clusters.keys())

        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                c1, c2 = cluster_ids[i], cluster_ids[j]

                d = linkage_distance(
                    clusters[c1],
                    clusters[c2],
                    dist_matrix,
                    data,
                    linkage
                )

                if d < min_dist:
                    min_dist = d
                    pair_to_merge = (c1, c2)

        c1, c2 = pair_to_merge

        new_cluster = clusters[c1] + clusters[c2]
        clusters[next_cluster_id] = new_cluster

        merges.append([c1, c2, min_dist, len(new_cluster)])

        del clusters[c1]
        del clusters[c2]

        next_cluster_id += 1

    return np.array(merges)
