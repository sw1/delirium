pacman::p_load(text2vec,stopwords,tidyverse,tm,glue)

# script to create document embeddings from word embeddings

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

a <- 1e-5 # doc embedding weighting parameter

cat('Reading table.\n')
dat <- read_csv(file.path(path,'to_python',
                          'tbl.csv.gz')) %>%
  mutate(hpi_hc=str_squish(str_replace_all(hpi_hc,'[[:punct:]]',''))) %>%
  select(id,set,text=hpi_hc)

vocab <- read_rds(
  file.path(path,'data_out',glue('12_vocab.rds')))

# create word downweighting for training only
freqs <- vocab$term_count/sum(vocab$term_count) 
w <- a/(a+freqs)

# loop through both tables and build doc vectors
for (s in unique(dat$set)){
  
  # downweight word vectors
  word_vectors <- read_rds(
    file.path(path,'data_out','13_word_vectors.rds'))
  word_vectors <- word_vectors * w
  
  cat(glue('\nBuilding training doc vectors for {s}.\n\n'))
  dtm <- read_rds(file.path(path,'data_out',glue('12_dtm_{s}.rds')))
  doc_vectors <- as.matrix((dtm %*% word_vectors)/Matrix::rowSums(dtm))
  rownames(doc_vectors) <- rownames(dtm)
  
  cat(glue('\nSaving output for {s}.\n\n'))
  write_rds(doc_vectors,file.path(path,'data_out',
                                  glue('14_doc_vectors_{s}.rds')))
  
}




