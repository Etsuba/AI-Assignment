import numpy as np

def compute_dist_matrix(data):
    """
    Compute full pairwise Euclidean distance matrix for data.
    data: (n, d) array
    Returns: (n, n) distance matrix
    """
    # Efficient: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    squared_norms = np.sum(data**2, axis=1, keepdims=True)  # (n,1)
    dist_sq = squared_norms + squared_norms.T - 2 * (data @ data.T)
    np.maximum(dist_sq, 0, out=dist_sq)  # numerical safety
    return np.sqrt(dist_sq)

def cluster_distance(group1, group2, dist_matrix, method="average"):
    """
    Compute distance between two clusters given base pairwise matrix.
    group1, group2: lists of point indices
    dist_matrix: (n, n) distance matrix
    method: 'single' | 'complete' | 'average'
    """
    idx1 = np.array(group1)
    idx2 = np.array(group2)
    submatrix = dist_matrix[np.ix_(idx1, idx2)]
    if method == "single":
        return np.min(submatrix)
    elif method == "complete":
        return np.max(submatrix)
    elif method == "average":
        return np.mean(submatrix)
    else:
        raise ValueError("Unsupported linkage: " + str(method))

def hierarchical_clustering(data, num_clusters=2, method="average"):
    """
    Bottom-up agglomerative clustering.
    data: (n, d) array
    num_clusters: desired number of clusters (1 <= num_clusters <= n)
    method: 'single' | 'complete' | 'average'
    Returns: list of clusters (each cluster is a list of point indices),
             and np.array of labels (n,)
    """
    n_samples = data.shape[0]
    if num_clusters < 1 or num_clusters > n_samples:
        raise ValueError("num_clusters must satisfy 1 <= num_clusters <= n")

    # Start with singleton clusters
    groups = [[i] for i in range(n_samples)]
    dist_matrix = compute_dist_matrix(data)

    # Repeatedly merge two closest clusters
    while len(groups) > num_clusters:
        best_val = np.inf
        best_pair = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d_val = cluster_distance(groups[i], groups[j], dist_matrix, method)
                # Deterministic tie-breaking using indices
                if (d_val < best_val) or (
                    d_val == best_val and best_pair is not None and (i, j) < best_pair
                ):
                    best_val = d_val
                    best_pair = (i, j)
        # Merge the best pair
        i, j = best_pair
        groups[i].extend(groups[j])
        del groups[j]

    # Produce labels
    labels = np.empty(n_samples, dtype=int)
    for cid, grp in enumerate(groups):
        for idx in grp:
            labels[idx] = cid

    return groups, labels

if __name__ == "__main__":
    # Example: synthetic 2D data with three groups
    rng = np.random.default_rng(42)
    cluster1 = rng.normal(loc=(2.0, 2.0), scale=0.3, size=(20, 2))
    cluster2 = rng.normal(loc=(6.0, 6.0), scale=0.3, size=(20, 2))
    cluster3 = rng.normal(loc=(2.0, 6.0), scale=0.3, size=(20, 2))
    dataset = np.vstack([cluster1, cluster2, cluster3])

    groups, labels = hierarchical_clustering(dataset, num_clusters=3, method="average")
    print("Cluster sizes:", [len(g) for g in groups])
    print("Labels (first 10):", labels[:10])