import glob

import numpy as np


files = glob.glob("*.npz")
print("Summarizing correlations from these files:", " ".join(files))

corrs = dict()
for fn in files:
    f = np.load(fn)
    for k, v in f.items():
        if not k.endswith('_corr'):
            continue

        v = v[0]
        if k in corrs:
            corrs[k].append(v)
        else:
            corrs[k] = [v]

for c, v in corrs.items():
    print(f"{c}={v}")

for c, v in corrs.items():
    count = len(v)
    mean = np.mean(v)
    std = np.std(v)
    median = np.median(v)
    print(f"{c}: {count=} {mean=} {std=} {median=}")
