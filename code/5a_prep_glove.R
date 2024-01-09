library(text2vec)
library(stopwords)
library(tidyverse)
library(tm)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}

cat('Reading table.\n')
dat <- read_csv(file.path(path,'to_python','tbl_to_python_updated.csv.gz')) %>%
  mutate(hpi_hc=stripWhitespace(trimws(str_replace_all(hpi_hc,'[[:punct:]]',''),'both')))

train <- dat %>% 
  filter(set %in% c('train','val')) %>% 
  select(id,text=hpi_hc)
test <- dat %>% 
  select(id,text=hpi_hc) %>%
  anti_join(train,by='id')

tbls <- list(train=train,test=test)
rm(list=c('dat','train','test'))

cat('Creating vocab.\n')
tokenizer <- word_tokenizer(tbls[['train']]$text) 
it <- itoken(tokenizer,n_chunks=1,progresbar=TRUE,
             ids=as.character(tbls[['train']]$id))
vocab <- create_vocabulary(it,stopwords=stopwords::stopwords('en'),
                           ngram=c(ngram_min=1L,ngram_max=2L))
vocab <- prune_vocabulary(vocab,term_count_min=20L,
                          doc_proportion_min=4e-4,doc_proportion_max=1.0)
vectorizer <- vocab_vectorizer(vocab)
cat(sprintf('Vocab size: %s\n',print(nrow(vocab))))
write_rds(vocab,file.path(path,'data_in','vocab.rds'))

for (i in seq_along(tbls)){
  
  cat(sprintf('Creating tokenizer for %s.\n',names(tbls)[i]))
  tokenizer <- word_tokenizer(tbls[[i]]$text) 
  it <- itoken(tokenizer,n_chunks=1,progresbar=TRUE,ids=tbls[[i]]$id)
  
  cat(sprintf('Creating TCM and DTM for %s.\n',names(tbls)[i]))
  tcm <- create_tcm(it,vectorizer,skip_grams_window=8L)
  dtm <- create_dtm(it,vectorizer)
  cat(sprintf('DTM size for %s: %d x %d\n',names(tbls)[i],nrow(dtm),ncol(dtm)))
  
  cat(sprintf('Saving output for %s.\n',names(tbls)[i]))
  write_rds(tcm,file.path(path,'data_in',sprintf('tcm_%s.rds',names(tbls)[i])))
  write_rds(dtm,file.path(path,'data_in',sprintf('dtm_%s.rds',names(tbls)[i])))
  
}
