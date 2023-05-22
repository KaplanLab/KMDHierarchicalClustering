import numpy as np
import time

from KMDHierarchicalClustering import cluster_scoring
from KMDHierarchicalClustering.KMDAlgo import KMDClustering

pca = np.load('1Msim/1M_pca.npy')
#y_true = np.load('1Msim/1M_y_true.npy')
k = 50

t = time.time()
kmd_cluster = KMDClustering(k=k, min_cluster_size = 'compute', affinity = 'correlation', n_clusters=8, certainty=0.5)
kmd_cluster.fit(pca, sub_sample=True, percent_size=5000 )
clustering_time = time.time() - t
print("Clustering time:", clustering_time)

all_idx_to_assign = kmd_cluster.idx_to_assign
y_pred = kmd_cluster.predict(pca)
hang_time = time.time()-t+clustering_time
print("Hang time:", hang_time)

np.save("1Msim/1M_y_pred.npy", y_pred)
np.save("1Msim/1M_idx_sampled.npy", kmd_cluster.idx_sampled)
