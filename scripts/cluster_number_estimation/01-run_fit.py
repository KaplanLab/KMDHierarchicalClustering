import sys

import numpy as np

from KMDHierarchicalClustering.KMDAlgo import KMDClustering


k_range = np.arange(1, 100)
min_cluster_size = 10
affinity='correlation'

infile = sys.argv[1]
outfile = sys.argv[2]

print(f"Reading data from {infile}")
npz = np.load(infile)
X = npz['X']

print("Creating linkages with {affinity=} {min_cluster_size=} {k_range=}")
kmd = KMDClustering(k='compute', k_range=k_range, affinity=affinity)
kmd.fit(X, keep_Z_list=True)
Z = kmd.Z_list
dists = kmd.dists

print("Saving fit file to", outfile)
np.savez(outfile, Z=Z, dists=dists, k_range=k_range, mcs=min_cluster_size)
