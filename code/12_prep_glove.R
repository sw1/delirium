pacman::p_load(text2vec,stopwords,tidyverse,tm,glue)

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
                          'tbl_to_python_expertupdate_chunked.csv.gz')) %>%
  mutate(hpi_hc=str_squish(str_replace_all(hpi_hc,'[[:punct:]]','')))

# create training set which will include only train set since
# longformer finetuning will only train on training (uses validation for
# longformer eval but doesnt train on). 
# the remainder will include val and all test sets which can be classified at
# the end and split into corresponding test sets after classification
train <- dat %>% 
  filter(set %in% c('train')) %>% 
  select(id,text=hpi_hc)
test <- dat %>% 
  select(id,text=hpi_hc) %>%
  anti_join(train,by='id')

# now tables for expert only
cat('Reading table.\n')
dat <- read_csv(
  file.path(path,'to_python',
            'tbl_to_python_expertupdate_onlyexpert_chunked.csv.gz')) %>%
  mutate(hpi_hc=str_squish(str_replace_all(hpi_hc,'[[:punct:]]','')))

train_onlyexpert <- dat %>%
  filter(set == 'train') %>%
  select(id,text=hpi_hc)
test_onlyexpert <- dat %>%
  filter(set == 'heldout_expert') %>%
  select(id,text=hpi_hc)

# now tables for full expert
cat('Reading table.\n')
dat <- read_csv(
  file.path(path,'to_python',
            'tbl_to_python_expertupdate_fullexpert_chunked.csv.gz')) %>%
  mutate(hpi_hc=str_squish(str_replace_all(hpi_hc,'[[:punct:]]','')))

train_fullexpert <- dat %>%
  filter(set == 'train') %>%
  select(id,text=hpi_hc)
test_fullexpert <- dat %>%
  filter(set == 'heldout_expert') %>%
  select(id,text=hpi_hc)

tbls <- list(train=train,test=test,
             train_onlyexpert=train_onlyexpert,
             test_onlyexpert=test_onlyexpert,
             train_fullexpert=train_fullexpert,
             test_fullexpert=test_fullexpert)
rm(list=c('dat','train','test'))

# save tables for doc embedding
write_rds(tbls,file.path(path,'data_in','glove_tbls.rds'))

# create vocab from train set and prune, remove stopwords, then save
fns_train <- names(tbls[str_detect(names(tbls),'train')])
for (fn in fns_train){
  
  cat(glue('\n\nCreating vocab for {fn}.\n\n'))
  tokenizer <- word_tokenizer(tbls[[fn]]$text) 
  it <- itoken(tokenizer,n_chunks=1,progresbar=TRUE,
               ids=as.character(tbls[[fn]]$id))
  vocab <- create_vocabulary(it,stopwords=stopwords::stopwords('en'),
                             ngram=c(ngram_min=1L,ngram_max=2L))
  vocab <- prune_vocabulary(vocab,term_count_min=20L,
                            doc_proportion_min=4e-4,doc_proportion_max=1.0)
  
  cat(glue('\nVocab size: {nrow(vocab)}\n\n'))
  write_rds(vocab,file.path(path,'data_in',glue('vocab_{fn}.rds')))

}

# create tcm and dtm for training and testing tables
for (fn in names(tbls)){
  
  if (str_detect(fn,'onlyexpert')){
    suffix <- 'train_onlyexpert'
  }else if (str_detect(fn,'fullexpert')){
    suffix <- 'train_fullexpert'
  }else{
    suffix <- 'train'
  }
  
  vocab <- read_rds(file.path(path,'data_in',glue('vocab_{suffix}.rds')))
  vectorizer <- vocab_vectorizer(vocab)
  
  cat(glue('\n\nCreating tokenizer for {fn}.\n\n'))
  tokenizer <- word_tokenizer(tbls[[fn]]$text) 
  it <- itoken(tokenizer,n_chunks=1,progresbar=TRUE,ids=tbls[[fn]]$id)
  
  cat(glue('\nCreating TCM and DTM for {fn}.\n\n'))
  tcm <- create_tcm(it,vectorizer,skip_grams_window=8L)
  dtm <- create_dtm(it,vectorizer)
  cat(glue('\nDTM size for {fn}: {nrow(dtm)} x {ncol(dtm)}\n\n'))
  
  cat(glue('Saving output for {fn}.\n\n'))
  write_rds(tcm,file.path(path,'data_in',glue('tcm_{fn}.rds')))
  write_rds(dtm,file.path(path,'data_in',glue('dtm_{fn}.rds')))
  
}
