import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import scanpy as sc
import SCCAF

from KMDHierarchicalClustering import cluster_scoring


parser = argparse.ArgumentParser()
parser.add_argument('--seed', help='seed for random selection', type=int, default=0)
parser.add_argument('--no-save', help='dont save to file', action='store_false', dest='save', default=True)
parser.add_argument('--algorithm', help='which algorithm to use', type=str, choices=['louvain', 'leiden', 'sccaf'], required=True)
args = parser.parse_args()

pca = pd.read_csv('./pure6_pca.csv', index_col=0)
y_true = pca['label'].to_numpy()
del pca['label']
X = pca.to_numpy()
n_clusters = np.unique(y_true).size

adata = sc.AnnData(X, obs=dict(labels=y_true))
print(f"Clustering {X.shape[0]} samples with {args.algorithm}")
s_time = time.time()
sc.pp.neighbors(adata, n_neighbors=15)
if args.algorithm != 'sccaf':
    alg = getattr(sc.tl, args.algorithm)
    alg(adata, resolution=1, random_state=args.seed)
else:
    adata.raw = adata
    sc.tl.louvain(adata, resolution=1, random_state=args.seed)
    adata.obs['L1_Round0'] = adata.obs['louvain']
    sccaf = []
    for i in range(10):
        SCCAF.SCCAF_optimize_all(adata, prefix='L1',  plot=False)
        sccaf.append(adata.obs['L1_result'])

e_time = time.time()
duration = e_time - s_time
print(f'Done. Took {duration} seconds')

if args.algorithm != 'sccaf':
    y_pred = adata.obs[args.algorithm].cat.codes.to_numpy()

    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc = cluster_scoring.hungarian_acc(y_true, y_pred)[0]
    print(f"Scores: {nmi=} {ari=} {acc=}")
else:
    nmi_list = []
    ari_list = []
    acc_list = []
    for y_pred in sccaf:
        y_pred = y_pred.astype(int).to_numpy()
        nmi_list.append(normalized_mutual_info_score(y_true, y_pred))
        ari_list.append(adjusted_rand_score(y_true, y_pred))
        acc_list.append(cluster_scoring.hungarian_acc(y_true, y_pred)[0])
    nmi = (np.mean(nmi_list), np.std(nmi_list))
    ari = (np.mean(ari_list), np.std(ari_list))
    acc = (np.mean(acc_list), np.std(acc_list))
    print(f"Scores: nmi={nmi[0]:.3f}+-{nmi[1]:.3f} ari={ari[0]:.3f}+-{ari[1]:.3f} acc={acc[0]:.3f}+-{acc[1]:.3f}")

if args.save:
    output_filename = f"pure6_{args.algorithm}_seed{args.seed}.npz"
    print("Saving to", output_filename)
    np.savez(output_filename, seed=args.seed, y_pred=y_pred, nmi_pred=nmi, ari_pred=ari,
             acc_pred=acc, duration=duration)
