args = commandArgs(trailingOnly=T)
if (length(args) == 0) {
	stop("Usage: 02-cluster.R CSV_FILE")
}

library(Seurat)
library(aricode)
library(stringr)

filename = args[1]
cat(sprintf("Reading from %s\n", filename))
df = read.csv(filename)
X = t(df[, c(1, 2)])
y = df[, 3]

s = CreateSeuratObject(X)
#s = NormalizeData(s)
s = ScaleData(s)
s = RunPCA(s, features = c("x1", "x2"), approx=F, npcs=2)
s = FindNeighbors(s, dims=2)
s = FindClusters(s)

output = str_replace(filename, ".csv$", "_seurat.rds")
cat(sprintf("Saving Seurat object to %s\n", output))
saveRDS(s, output)

output = str_replace(filename, ".csv$", "_pred_seurat.csv")
y_pred = Idents(s)
cat(sprintf("Saving labels to %s\n", output))
write.csv(y_pred, output)

ari = ARI(y, y_pred)

cat("ARI=")
cat(ari)
cat("\n")
