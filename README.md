# KMD clustering
A clustering method based on hierarchical clustering using KMD linkage, KMD silhouette and outlier-aware partitioning

 ## Requirements 
Python: tested on versions 3.6.5 and 3.7. We don't expect this to work with Python 2.x

Python Dependencies:
```
 numpy>=1.18.4
 scipy>=1.3.0
 sklearn>=0.23.1
 matplotlib>=3.1.2
 itertools>=8.5.0
 ```

## Getting started
Download the project to your work directory:
```
git clone https://github.com/KMDHierarchicalClustering/KMDHierarchicalClustering.git
```

## Usage in Python environment 

```
from KMDHierarchicalClustering.KMDclustering.KMDAlgo import KMDLinkage
```

- Parameters to specify:

  X: dataset to cluster
  
  k: number of minimum distances used to calculate distance between clusters. if set to is 'compute' (default), KMD silhouette will be used to choose k.
  
  n_clusters: number of clusters
  
  min_cluster_size: minimal cluster size. If a cluster is smaller then this size it is considered to be an outlier.
  
  - Parameters that are recommended to be used as default:

    affinity: metric used to compute the distance. Can be "euclidean" (default), "correlation", "spearman","precomputed"
    or any metric used by `scipy.spatial.distance.pdist`. If "precomputed", an adjacency matrix is given as input. 
    
    certainty: number between 0.5 and 1 indicating a the confidence threshold to use for outlier assignment to core clusters. 0.5 means all outliers will be left out (no assigment), 1 means all outliers will be assigned to core clusters.

    k_scan_range: tuple indicating the range of k values to be used to search for k. Given as (start k, stop k, jumps).

    y_true: True cluster labels (only used in evaluation scenario).

    plot_scores: if True, a plot of KMD Silhouette vs accuracy across different k values will be generated (only used in evaluation scenario).

    path: Path to self prediction for each k. If False, prediction will not be saved will be required.
 
## Usage example

```
kmd_cluster = KMDLinkage(k='compute', n_clusters = 2, min_cluster_size = 10)
kmd_cluster.fit(X)
y = kmd_cluster.predict(X)

```

## demos 
Toy_dataset_example.ipynb - Evaluates our method’s performance on a standard set of simulated toy datasets - expected run time: ~25 minutes

Zeisel15_dataset.ipynb - Example of method’s performance on a single cell dataset - - expected run time: ~8 minutes

Lawlor17_dataset.ipynb - Example of method’s performance on a single cell dataset - - expected run time: ~2 minutes

Li17_dataset.ipynb - Example of method’s performance on a single cell dataset - - expected run time: ~1 minute

