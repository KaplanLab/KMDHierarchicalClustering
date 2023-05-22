import sys
import re

import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from KMDHierarchicalClustering import cluster_scoring

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} CSV_FILE1 CSV_FILE2 ...")
    print("CSV_FILE should have the format DATASET_pred_X for some X and some DATASET.npz file")
    sys.exit(1)

files = sys.argv[1:]
ds_names = [ re.sub("_pred.*", "", x) for x in files ]
ds_datafiles = [ f"{x}.npz" for x in ds_names ]
results = dict(accuracy = [], nmi=[], ari=[])
#score_names = ["accuracy", "nmi", "ari"]
#index = pd.MultiIndex.from_product([ds_names, score_names])
#results = np.empty(len(ds_names), len(score_names))

for y_file, ds_file in zip(files, ds_datafiles):
    y_pred = pd.read_csv(y_file, index_col=0).to_numpy().squeeze()
    y_true = np.load(ds_file)['y']
    acc = cluster_scoring.hungarian_acc(y_true, y_pred)[0]
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    results['accuracy'].append(acc)
    results['nmi'].append(nmi)
    results['ari'].append(ari)

df = pd.DataFrame(results, index=ds_names)
print(df)
