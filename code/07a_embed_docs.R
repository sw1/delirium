library(text2vec)
library(stopwords)
library(tidyverse)
library(tm)

a <- 1e-5

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

vocab <- read_rds(file.path(path,'data_in','vocab.rds'))
freqs <- vocab$term_count/sum(vocab$term_count) # weighting by f in train only
w <- a/(a+freqs)

word_vectors <- read_rds(file.path(path,'data_in','word_vectors.rds'))
word_vectors <- word_vectors * w

for (fn in names(tbls)){
  
  cat(sprintf('Building training doc vectors for %s.\n',fn))
  dtm <- read_rds(file.path(path,'data_in',sprintf('dtm_%s.rds',fn)))
  doc_vectors <- as.matrix((dtm %*% word_vectors)/Matrix::rowSums(dtm))
  rownames(doc_vectors) <- rownames(dtm)
  
  cat(sprintf('Saving output for %s.\n',fn))
  write_rds(doc_vectors,file.path(path,'data_in',sprintf('doc_vectors_%s.rds',fn)))
  
}




