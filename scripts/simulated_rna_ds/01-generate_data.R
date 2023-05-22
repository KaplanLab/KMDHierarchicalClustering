# Adapted from: https://github.com/willtownes/glmpca/blob/master/vignettes/glmpca.Rmd
require(glmpca)

set.seed(202)
ngenes <- 5000 #must be divisible by 10
ngenes_informative<-ngenes*.1
ncells <- 50 #number of cells per cluster, must be divisible by 2
nclust<- 3
# simulate two batches with different depths
batch<-rep(1:2, each = nclust*ncells/2)
ncounts <- rpois(ncells*nclust, lambda = 1000*batch)
# generate profiles for 3 clusters
profiles_informative <- replicate(nclust, exp(rnorm(ngenes_informative)))
profiles_const<-matrix(ncol=nclust,rep(exp(rnorm(ngenes-ngenes_informative)),nclust))
profiles <- rbind(profiles_informative,profiles_const)
# generate cluster labels
label <- sample(rep(1:3, each = ncells))
# generate single-cell transcriptomes 
counts <- sapply(seq_along(label), function(i){
					rmultinom(1, ncounts[i], prob = profiles[,label[i]])
})
rownames(counts) <- paste("gene", seq(nrow(counts)), sep = "_")
colnames(counts) <- paste("cell", seq(ncol(counts)), sep = "_")
# clean up rows
Y <- counts[rowSums(counts) > 0, ]
# Combine to one matrix
ds = rbind(Y, label)
# Save as CSV, with cells as rows
write.csv(t(ds), "generated_data.csv")
