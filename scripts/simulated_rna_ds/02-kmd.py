import argparse
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from KMDHierarchicalClustering import KMDClustering, cluster_scoring


parser = argparse.ArgumentParser()
#parser.add_argument('--size',help='number of samples to downsample', type=int, default=5000)
parser.add_argument('--seed', help='seed for random selection', type=int, default=1)
args = parser.parse_args()

ds = pd.read_csv('./generated_data.csv', index_col=0)
y_true = ds['label']
del ds['label']
X = ds.to_numpy()
n_clusters = np.unique(y_true).size

print(f"Clustering {X.shape[0]} samples")
random.seed(args.seed)
s_time = time.time()
kmd_cluster=KMDClustering(k='compute', min_cluster_size='compute', affinity='correlation',
                          n_clusters=n_clusters, certainty=0.5, k_scan_range=range(1, 100, 5))
kmd_cluster.fit(X)
y_pred = kmd_cluster.predict(X)

e_time = time.time()
duration = e_time - s_time
print(f'Done. Took {duration} seconds')


nmi_pred = normalized_mutual_info_score(y_true, y_pred)
ari_pred = adjusted_rand_score(y_true, y_pred)
acc_pred = cluster_scoring.hungarian_acc(y_true, y_pred)[0]
print(f"Result (full): nmi={nmi_pred} ari={ari_pred} acc={acc_pred}")
