pacman::p_load(text2vec,stopwords,tidyverse,doParallel,glue,tm)


if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
  all_cores <- 4
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
  all_cores <- 14
}
source(file.path(path,'code','fxns.R'))

tbl_name <- 'tbl_allnotes.csv.gz'

# read table, remove punctuation, and squish whitespace
cat('Reading table.\n')
dat <- read_csv(file.path(path,'to_python',tbl_name)) %>%
  mutate(hpi_hc=str_squish(str_replace_all(hpi_hc,'[[:punct:]]',''))) %>%
  select(id,set,text=hpi_hc)

it <- word_tokenizer(dat %>% filter(set == 'train') %>% pull(text)) %>%
  itoken(n_chunks=1,progresbar=TRUE,
         ids=dat %>% filter(set == 'train') %>%
           pull(id) %>% as.character())

vocab <- create_vocabulary(it,stopwords=stopwords::stopwords('en'),
                           ngram=c(ngram_min=1L,ngram_max=2L)) %>%
  prune_vocabulary(term_count_min=200L,
                   doc_proportion_min=8e-4,doc_proportion_max=1.0)

cat(glue('\nVocab size: {nrow(vocab)}\n\n'))
write_rds(vocab,file.path(path,'data_out','12_lam_vocab.rds'))

vectorizer <- vocab_vectorizer(vocab)

for (s in unique(dat$set)){

  it <- word_tokenizer(dat %>% filter(set == s) %>% pull(text)) %>%
    itoken(n_chunks=1,progresbar=TRUE,
           ids=dat %>% filter(set == s) %>%
             pull(id) %>% as.character())

  cat(glue('\nCreating TCM and DTM for {s}.\n\n'))
  tcm <- create_tcm(it,vectorizer,skip_grams_window=8L)
  dtm <- create_dtm(it,vectorizer)
  cat(glue('\nDTM size for {s}: {nrow(dtm)} x {ncol(dtm)}\n\n'))

  cat(glue('Saving output for {s}.\n\n'))
  write_rds(tcm,file.path(path,'data_out',glue('12_lam_tcm_{s}.rds')))
  write_rds(dtm,file.path(path,'data_out',glue('12_lam_dtm_{s}.rds')))

}

rm(list=c('it','tcm','dtm','dat','vocab','vectorizer'))


#all_cores <- parallel::detectCores(logical=FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

tcm <- readRDS(file.path(path,'data_out','12_lam_tcm_train.rds'))

set.seed(123)
glove <- GlobalVectors$new(rank=128,x_max=10,lambda=1e-5,learning_rate=0.1)
wv_main <- glove$fit_transform(tcm,n_iter=25,convergence_tol=0.01,
                               n_threads=all_cores)
wv_context <- glove$components
word_vectors <- wv_main + t(wv_context)

write_rds(word_vectors,
          file.path(path,'data_out',
                    glue('13_lam_word_vectors.rds')))
write_rds(wv_main,
          file.path(path,'data_out',
                    glue('13_lam_w2v.rds')))

stopCluster(cl)

rm(list=c('glove','wv_main','wv_context','word_vectors'))

a <- 1e-5 # doc embedding weighting parameter

cat('Reading table.\n')
dat <- read_csv(file.path(path,'to_python',tbl_name)) %>%
  mutate(hpi_hc=str_squish(str_replace_all(hpi_hc,'[[:punct:]]',''))) %>%
  select(id,set,text=hpi_hc)

vocab <- read_rds(
  file.path(path,'data_out',glue('12_lam_vocab.rds')))

# create word downweighting for training only
freqs <- vocab$term_count/sum(vocab$term_count) 
w <- a/(a+freqs)

# loop through both tables and build doc vectors
for (s in unique(dat$set)){
  
  # downweight word vectors
  word_vectors <- read_rds(
    file.path(path,'data_out','13_lam_word_vectors.rds'))
  word_vectors <- word_vectors * w
  
  cat(glue('\nBuilding training doc vectors for {s}.\n\n'))
  dtm <- read_rds(file.path(path,'data_out',glue('12_lam_dtm_{s}.rds')))
  doc_vectors <- as.matrix((dtm %*% word_vectors)/Matrix::rowSums(dtm))
  rownames(doc_vectors) <- rownames(dtm)
  
  cat(glue('\nSaving output for {s}.\n\n'))
  write_rds(doc_vectors,file.path(path,'data_out',
                                  glue('14_lam_doc_vectors_{s}.rds')))
  
}


