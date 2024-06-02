pacman::p_load(text2vec,stopwords,tidyverse,tm,glue)

# script to create document embeddings from word embeddings

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

a <- 1e-5 # doc embedding weighting parameter

# read preprocessed tables, vocab
tbls <- read_rds(file.path(path,'data_in','glove_tbls.rds'))

# loop through both tables and build doc vectors
for (fn in names(tbls)){
  
  if (str_detect(fn,'onlyexpert')){
    suffix <- 'train_onlyexpert' 
  }else if (str_detect(fn,'fullexpert')){
    suffix <- 'train_fullexpert'
  }else{
    suffix <- 'train'
  }
  
  vocab <- read_rds(
    file.path(path,'data_in',glue('vocab_{suffix}.rds')))
  
  # create word downweighting for training only
  freqs <- vocab$term_count/sum(vocab$term_count) 
  w <- a/(a+freqs)
  
  # downweight word vectors
  word_vectors <- read_rds(
    file.path(path,'data_in',glue('word_vectors_{suffix}.rds')))
  word_vectors <- word_vectors * w
  
  cat(glue('\nBuilding training doc vectors for {fn}.\n\n'))
  dtm <- read_rds(file.path(path,'data_in',glue('dtm_{fn}.rds')))
  doc_vectors <- as.matrix((dtm %*% word_vectors)/Matrix::rowSums(dtm))
  rownames(doc_vectors) <- rownames(dtm)
  
  cat(glue('\nSaving output for {fn}.\n\n'))
  write_rds(doc_vectors,file.path(path,'data_in',glue('doc_vectors_{fn}.rds')))
  
}




