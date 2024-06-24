pacman::p_load(text2vec,stopwords,tidyverse,doParallel,tm,glue)

# script to create vocab, dtm, and tcm for glove

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

# read table, remove punctuation, and squish whitespace
cat('Reading table.\n')
dat <- read_csv(file.path(path,'to_python',
                          'tbl.csv.gz')) %>%
  mutate(hpi_hc=str_squish(str_replace_all(hpi_hc,'[[:punct:]]',''))) %>%
  select(id,set,text=hpi_hc)

it <- word_tokenizer(dat %>% filter(set == 'train') %>% pull(text)) %>%
  itoken(n_chunks=1,progresbar=TRUE,
         ids=dat %>% filter(set == 'train') %>% 
           pull(id) %>% as.character())

vocab <- create_vocabulary(it,stopwords=stopwords::stopwords('en'),
                           ngram=c(ngram_min=1L,ngram_max=2L)) %>%
  prune_vocabulary(term_count_min=20L,
                   doc_proportion_min=4e-4,doc_proportion_max=1.0)

cat(glue('\nVocab size: {nrow(vocab)}\n\n'))
write_rds(vocab,file.path(path,'data_out','12_vocab.rds'))

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
  write_rds(tcm,file.path(path,'data_out',glue('12_tcm_{s}.rds')))
  write_rds(dtm,file.path(path,'data_out',glue('12_dtm_{s}.rds')))
  
}
