import argparse
import random
import glob

import numpy as np
from KMDHierarchicalClustering.predict_clust_label import normalize_kmd_silhouette

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='dataset name', required=True)
parser.add_argument('--seed', help='seed', required=True)
parser.add_argument('--debug', help='print debugging output', required=False, default=False, type=bool)
parser.add_argument('--results-dir', help='results dir', required=False, default='resluts', type=str)

args = parser.parse_args()

#parameters
ds_name = args.name
seed = args.seed
results_dir = args.results_dir
glob_pattern = f"{results_dir}/{ds_name}_sample20000_seed{seed}_k*_fit.npz"

filenames = glob.glob(glob_pattern)
fits = [ np.load(f) for f in filenames ]
k_range = np.array([ f['k'] for f in fits ])
sils = np.array([ f['kmd_sil'] for f in fits ])
n = fits[0]['y_true'].shape[0]

normalized_sils = normalize_kmd_silhouette(sils, k_range, n)
if args.debug:
    for fn, s in zip(filenames, normalized_sils):
        print(f"{fn=} {s=:.4f}")
best_fit = np.argmax(normalized_sils)
if args.debug:
    print("best:", filenames[best_fit])
else:
    print(filenames[best_fit])

