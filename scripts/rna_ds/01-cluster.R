args = commandArgs(trailingOnly=T)
if (length(args) == 0) {
	stop("Usage: 02-cluster.R CSV_FILE")
}

library(Seurat)
library(aricode)
library(stringr)
library(reticulate)

npy = import('numpy')

filename = args[1]
cat(sprintf("Reading from %s\n", filename))
npz = npy$load(filename) # Created from notebooks in repo
X = t(npz[['X']])
y = as.factor(npz[['y']])

colnames(X) = 1:dim(X)[2]
rownames(X) = 1:dim(X)[1]

s = CreateSeuratObject(X)
#s = NormalizeData(s) # Data is already pre-processed
s = ScaleData(s)
s = FindVariableFeatures(s) 
s = RunPCA(s) #, features=rownames(s))
s = FindNeighbors(s)
s = FindClusters(s)

output = str_replace(filename, ".npz$", "_seurat.rds")
cat(sprintf("Saving Seurat object to %s\n", output))
saveRDS(s, output)

output = str_replace(filename, ".npz$", "_pred_seurat.csv")
y_pred = Idents(s)
cat(sprintf("Saving labels to %s\n", output))
write.csv(y_pred, output)

ari = ARI(y, y_pred)

cat("ARI=")
cat(ari)
cat("\n")
