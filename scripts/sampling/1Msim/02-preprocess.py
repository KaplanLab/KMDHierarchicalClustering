import time

import scanpy as sc
import numpy as np
import pandas as pd
import anndata


def load_df(filename):
    with np.load(filename, allow_pickle=True) as f:
        obj = pd.DataFrame(**f)
    return obj

print("Loading data")
simulator_counts = load_df('1Msim/counts.npz')
simulator_counts = simulator_counts.astype(float) # Required for normalize_per_cell()
simulator_cellparams = load_df('1Msim/cellparams.npz')

print("Preprocessing")
adata = anndata.AnnData(simulator_counts)
adata.obs = simulator_cellparams
sc.pp.filter_genes(adata, min_counts=1)         # only consider genes with more than 1 count
sc.pp.normalize_per_cell(                       # normalize with total UMI count per cell
     adata, key_n_counts='n_counts_all')
filter_result = sc.pp.filter_genes_dispersion(  # select highly-variable genes
    adata.X, flavor='cell_ranger', n_top_genes=1000, log=False
)
adata = adata[:, filter_result.gene_subset]     # subset the genes
sc.pp.normalize_per_cell(adata)                 # renormalize after filtering
# sc.pp.log1p(adata)                      # log transform: adata.X = log(adata.X + 1)
sc.pp.scale(adata)
sc.tl.pca(adata)
y_true =  adata.obs['group']
pca = adata.obsm['X_pca']

print("Saving files")
np.save('1Msim/1M_y_true',y_true.astype(int))
np.save('1Msim/1M_pca',pca)
