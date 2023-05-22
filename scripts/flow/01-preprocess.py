import argparse
import random

import numpy as np
from FlowCytometryTools import FCMeasurement
from KMDHierarchicalClustering import KMDClustering

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='dataset name', required=True)
parser.add_argument('--seed', help='seed for random selection', type=int, required=True)
parser.add_argument('--sample-size', help='how much to sample', type=int, default=20000, required=False)
parser.add_argument('--ignore-cols', help='which columns to ignore in dataset (can be negative)', nargs='+', type=int, required=False)

args = parser.parse_args()

KNOWN_IGNORED_COLS = {
        'levine_13dim': [-1],
        'levine_32dim': np.r_[np.arange(4), np.arange(-5, 0)],
        'samusik_01': np.r_[np.arange(8), np.arange(-7, 0)],
        }

#parameters
ds_name = args.name
results_dir = 'preprocessed'
data_dir = 'raw_data'
datafile = f'{data_dir}/{ds_name}.fcs'
print("Working on dataset from", datafile)
sample = FCMeasurement(ID='Test Sample', datafile=datafile)
df = sample.data
orig_ds_shape = df.shape
print(f"Dataset has {orig_ds_shape[0]} rows before filtering")

valid_rows = ~df['label'].isna()
df = df[valid_rows]
ds_len = df.shape[0]
print(f"Dataset has {ds_len} rows after filtering")

if args.ignore_cols is not None:
    ignored_cols_idx = args.ignore_cols
else:
    ignored_cols_idx = KNOWN_IGNORED_COLS.get(ds_name.lower(), [-1])
print("Will ignore these columns for measurements:", ignored_cols_idx)

ignored_cols_idx = np.arange(df.shape[1])[ignored_cols_idx]
if df.shape[1]-1 not in ignored_cols_idx:
    print("WARNING: last column in ignore-cols. This is usually the label column and may produce weird results")

chosen_cols_idx = np.fromiter(set(np.arange(df.shape[1])) - set(ignored_cols_idx), dtype=int)

# randomly choose n cells from dataset
np.random.seed(args.seed)
idx = np.random.randint(ds_len, size=args.sample_size)
y_true = df['label'].astype(int).to_numpy()[idx]

data = df.to_numpy()[idx][:, chosen_cols_idx]
normalized = np.arcsinh(data / 5)

output_file = f'{results_dir}/{ds_name}_sample{args.sample_size}_seed{args.seed}.npz'
print("Saving to", output_file)
np.savez(output_file, X_unnormalized=data, X=normalized, y=y_true, seed=args.seed, original_shape=orig_ds_shape, ignored_cols_idx=ignored_cols_idx, sampled_cells=idx)
