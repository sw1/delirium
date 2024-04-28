pacman::p_load(text2vec,stopwords,tidyverse,doParallel)

# script to fit glove

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

all_cores <- parallel::detectCores(logical=FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

tcm <- readRDS(file.path(path,'data_in','tcm_train.rds'))

set.seed(123)
glove <- GlobalVectors$new(rank=128,x_max=10,lambda=1e-5,learning_rate=0.1)
wv_main <- glove$fit_transform(tcm,n_iter=25,convergence_tol=0.01,
                               n_threads=all_cores)
wv_context <- glove$components
word_vectors <- wv_main + t(wv_context)

write_rds(word_vectors,file.path(path,'data_in','word_vectors.rds'))
write_rds(wv_main,file.path(path,'data_in','w2v.rds'))

stopCluster(cl)
