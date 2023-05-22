import argparse
import time

import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import scanpy as sc

from KMDHierarchicalClustering import cluster_scoring


parser = argparse.ArgumentParser()
parser.add_argument('--seed', help='seed for random selection', type=int, default=0)
parser.add_argument('--algorithm', help='which algorithm to use', type=str, choices=['louvain', 'leiden'], required=True)
args = parser.parse_args()

ds = pd.read_csv('./generated_data.csv', index_col=0)
y_true = ds['label']
del ds['label']
X = ds.to_numpy()
n_clusters = np.unique(y_true).size

adata = sc.AnnData(X, obs=dict(labels=y_true))
print(f"Clustering {X.shape[0]} samples with {args.algorithm}")
s_time = time.time()
sc.pp.neighbors(adata, n_neighbors=15)
alg = getattr(sc.tl, args.algorithm)
alg(adata, resolution=1, random_state=args.seed)
e_time = time.time()
duration = e_time - s_time
print(f'Done. Took {duration} seconds')

y_pred = adata.obs[args.algorithm].cat.codes.to_numpy()

nmi = normalized_mutual_info_score(y_true, y_pred)
ari = adjusted_rand_score(y_true, y_pred)
acc = cluster_scoring.hungarian_acc(y_true, y_pred)[0]

print(f"Scores: {nmi=} {ari=} {acc=}")
