import argparse

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import scanpy as sc

from KMDHierarchicalClustering import cluster_scoring


parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Filename to calculate scores for')
args = parser.parse_args()

fit = np.load(args.filename)
y_true = fit['y_true']
y_pred = fit['y_pred']

nmi = normalized_mutual_info_score(y_true, y_pred)
ari = adjusted_rand_score(y_true, y_pred)
acc = cluster_scoring.hungarian_acc(y_true, y_pred)[0]

print(f"{nmi=} {ari=} {acc=}")
