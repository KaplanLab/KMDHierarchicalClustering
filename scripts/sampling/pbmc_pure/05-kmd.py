import argparse
import time

import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from KMDHierarchicalClustering import KMDClustering, cluster_scoring


parser = argparse.ArgumentParser()
parser.add_argument('--size',help='number of samples to downsample', type=int, default=5000)
parser.add_argument('--seed', help='seed for random selection', type=int, default=1)
parser.add_argument('--nclusters', help='override number of clusters', type=int, default=None)
parser.add_argument('--mcs', help='min cluster size', type=int, default=50)
args = parser.parse_args()

pca = pd.read_csv('./pure6_pca.csv', index_col=0)
y_true = pca['label'].to_numpy()
del pca['label']
X = pca.to_numpy()
if args.nclusters:
    n_clusters = args.nclusters
else:
    n_clusters = np.unique(y_true).size

print(f"Clustering {X.shape[0]} samples downsampled to {args.size} with seed {args.seed}")
s_time = time.time()
kmd_cluster=KMDClustering(k='compute', min_cluster_size=args.mcs, affinity='correlation',
                          n_clusters=n_clusters, certainty=0.5, k_scan_range=range(1, 100))
kmd_cluster.fit(X, sub_sample=True, percent_size=args.size, seed=args.seed)
y_pred = kmd_cluster.predict(X)
y_pred_sub = kmd_cluster.y_pred_sub
y_true_sub = y_true[kmd_cluster.idx_sampled]

e_time = time.time()
duration = e_time - s_time
print(f'Done. Took {duration} seconds')


nmi_pred = normalized_mutual_info_score(y_true, y_pred)
ari_pred = adjusted_rand_score(y_true, y_pred)
acc_pred = cluster_scoring.hungarian_acc(y_true, y_pred)[0]
print(f"Result (full): nmi={nmi_pred} ari={ari_pred} acc={acc_pred}")

nmi_sub = normalized_mutual_info_score(y_true_sub, y_pred_sub)
ari_sub = adjusted_rand_score(y_true_sub, y_pred_sub)
acc_sub = cluster_scoring.hungarian_acc(y_true_sub, y_pred_sub)[0]
print(f"Result (sampled): nmi={nmi_sub} ari={ari_sub} acc={acc_sub}")

output_filename = f"pure6_kmd_seed{args.seed}_sample{args.size}_mcs{args.mcs}"
if args.nclusters:
    output_filename += f"_n{n_clusters}"
z_list_filename = output_filename + "_Z.npy"
output_filename += ".npz"
if hasattr(kmd_cluster, 'Z_list'):
    print("Saving Z list to", z_list_filename)
    np.save(z_list_filename, kmd_cluster.Z_list)
print("Saving to", output_filename)
np.savez(output_filename, Z=kmd_cluster.Z, k=kmd_cluster.k, seed=args.seed, y_pred=y_pred,
         idx_sub=kmd_cluster.idx_sampled, nmi_pred=nmi_pred, ari_pred=ari_pred, acc_pred=acc_pred,
         nmi_sub=nmi_sub, ari_sub=ari_sub, acc_sub=acc_sub, duration=duration, mcs=args.mcs, n_clusters=n_clusters)
