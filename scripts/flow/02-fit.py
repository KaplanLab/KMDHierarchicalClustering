import os
import sys
import argparse
import random

import numpy as np
from FlowCytometryTools import FCMeasurement
from KMDHierarchicalClustering import KMDClustering

parser = argparse.ArgumentParser()
parser.add_argument('path', help='path to data file')
parser.add_argument('-k', help='k to run (for parallel k scanning)', type=int, required=False)
parser.add_argument('--no-overwrite', dest='overwrite', help='do not overwrite existing file', action='store_false')

args = parser.parse_args()

#parameters
results_dir = 'results.unnormalized'
output_basename = os.path.splitext(os.path.basename(args.path))[0]
output_file = f'{results_dir}/{output_basename}_k{args.k}_fit.npz'
if not args.overwrite and os.path.exists(output_file):
    print("Refusing to overwrite output file", output_file)
    sys.exit(0)
print("Working on dataset from", args.path)
print("Will save to", output_file)
npz = np.load(args.path)

X = npz['X_unnormalized']
y_true = npz['y']

n_clusters = np.unique(y_true).shape[0]
if args.k is None:
    print("Will scan for the best k")
    args.k = 'compute'
else:
    print(f"Running with prechosen k={args.k}")
kmd_cluster = KMDClustering(k=args.k, min_cluster_size=50, n_clusters=n_clusters, affinity='correlation')
kmd_cluster.fit(X)
random.seed(1)
y_pred = kmd_cluster.predict(X)

print("Saving to", output_file)
np.savez(output_file, y_true=y_true, y_pred=y_pred, k=kmd_cluster.k, Z=kmd_cluster.Z, kmd_sil=kmd_cluster.sil_score)

