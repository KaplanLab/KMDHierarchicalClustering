library('Matrix')
set.seed(1)

cat("Reading matrix\n")
pure = list()
pure$full = readRDS('all_pure_select_11.rds')
cat("Selecting populations\n")
chosen = c(2, 3, 4, 5, 6, 9)
pure$selected_raw = lapply(chosen, function(i) { pure$full[[i]]})
pure$selected = list()
pure$selected$matrix = Matrix::t(do.call(rbind, pure$selected_raw))
pure$selected$labels = unlist(lapply(1:length(chosen), function (i) { rep(i, dim(pure$selected_raw[[i]])[1])}))
cat("Saving selected matrix\n")
saveRDS(pure$selected, "selected_pure_6.rds")
