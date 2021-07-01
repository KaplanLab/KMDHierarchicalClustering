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
 psutil>=5.8.0  # recommended for Windows and macOS platform users, in order to check available memory 
 ```

## Getting started
Installation using pip (requires pip and git):
```
pip install git+https://github.com/KaplanLab/KMDHierarchicalClustering
```
Alternatively, you can download or clone the repository.

Prefix with `sudo` for global installation, alternatively use `pip install --user` for installation in your home directory.

## Usage in Python environment 

Usage follows scikit-learn interface:

```
from KMDHierarchicalClustering import KMDClustering
kmd_cluster = KMDClustering(n_clusters = 2)
kmd_cluster.fit(X) # X is the dataset matrix
y = kmd_cluster.predict(X)
```

Parameters:

```
KMDClustering(k='compute', n_clusters = 2, min_cluster_size = 'compute', affinity = 'euclidean', certainty = 0.5 ,k_scan_range = (1,100,3), y_true = [], plot_scores=False, path=False)
```

The main parameter that users should set is `n_clusters`. We recommend using default values for the rest of the parameters unless there is a specific reason to change these.
 
`k`: number of minimum distances to calculate distance between clusters. if flag is compute, best k will be predicted.

`n_clusters`: number of clusters.

`min_cluster_size`: minimal cluster size. If a cluster is smaller then this size it is considered to be an outlier.

`affinity`: metric used to compute the distance. Can be "euclidean" (default), "correlation", "spearman", "precomputed"
    or any metric used by `scipy.spatial.distance.pdist`. If "precomputed", an adjacency matrix is given as input.
    
`certainty`:number between 0.5 and 1 indicating a the confidence threshold to use for outlier assignment to core clusters. 0.5 means all outliers will be left out (no assigment), 1 means all outliers will be assigned to core clusters.step (0.5 means all outliers will be assigned to core clusters; 1 means no outliers will be assigned).

`k_scan_range`: tuple indicating the range of k values to be used to search for k. Given as (start k, stop k, jumps).

`y_true`: true cluster labels (only used in evaluation scenario).

`plot_scores`: if True, a plot of KMD Silhouette vs accuracy across different k values will be generated (only used in evaluation scenario).

`path`: path to self prediction for each k, if False prediction will not be saved.

```
fit(self,X,sub_sample=False,percent_size=0.2,seed = 1)
```
`X` : Dataset to cluster in numpy array format 

`sub_sample`: If True ,dataset will be subsampled and a post hoc cluster inffering method will be conducted, reccomended on large dataset 

`percent_size`: fraction of dataset to subsample(if percent_size<1)/ size of dataset to subsample(if percent_size>1)

`seed`: seed to randomly subsample 


## Demos 

Toy_dataset_example.ipynb - Evaluates performance on a standard set of simulated toy datasets - expected run time: ~25 minutes

Zeisel15_dataset.ipynb - Evaluaets performance on a single cell dataset - - expected run time: ~8 minutes

Lawlor17_dataset.ipynb - Evaluates performance on a single cell dataset - - expected run time: ~20 seconds

Li17_dataset.ipynb - Evaluates performance on a single cell dataset - - expected run time: ~10 seconds

