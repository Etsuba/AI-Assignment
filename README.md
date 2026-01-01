## Agglomerative Hierarchical Clustering in Python

A simple, from-scratch implementation of **agglomerative hierarchical clustering** using Python and NumPy. Supports multiple linkage methods and generates dendrograms to visualize cluster hierarchies.


## Features
- Implements single, complete, average, and Ward linkage.
- Computes cluster labels and sizes.
- Generates dendrogram plots for visual analysis.
- No high-level ML libraries required.

## Installation

1. Clone the repository:
```bash
 https://github.com/Etsuba/AI-Assignment.git
````
2. Install dependencies:

```bash
pip install numpy matplotlib scipy

```

## Usage

Run the main script to generate synthetic data and perform clustering:

```bash
python hierarchical-clustering.py
```

Change the linkage method by setting the `linkage` parameter:

```python
merges = agglomerative_hierarchical_clustering(dataset, linkage="ward")
```

Outputs include **cluster labels**.

---

## References

1. Jain, A. K., Murty, M. N., & Flynn, P. J. (1999). *Data clustering: A review.* ACM Computing Surveys, 31(3), 264–323.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2nd ed.). Springer.
3. Ward, J. H. (1963). *Hierarchical grouping to optimize an objective function.* Journal of the American Statistical Association, 58(301), 236–244.
4. SciPy Hierarchical Clustering Documentation: [https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)

