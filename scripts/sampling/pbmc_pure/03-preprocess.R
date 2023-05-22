library('Seurat')

pure = readRDS('./selected_pure_6.rds')
s = CreateSeuratObject(pure$matrix)
s = NormalizeData(s, scale.factor=10000)
s = FindVariableFeatures(s, nfeatures=1000)
s = ScaleData(s)
s = RunPCA(s, npcs=20)
cat("Saving feature matrix and Seurat object\n")
mat = Embeddings(object = s, reduction = "pca")
label = pure$labels
mat_label = cbind(mat, label)
write.csv(mat_label, 'pure6_pca.csv')
saveRDS(s, "pure6_seurat.rds")
