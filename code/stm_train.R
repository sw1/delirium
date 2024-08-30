pacman::p_load(tidyverse,glue,gtsummary,flextable,icd.data,tidymodels,
               rpart,rpart.plot,officer,gridExtra,stm,LDAvis)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

ths <- c('70','80','90')
K <- 30

for (th in ths){
  run <- glue('fkw1_th{th}_fr100_pl1_lr2.0e-06_wd1.0e-01',
              '_nb16_ls0.0e+00_cw5_labpseudo_train.csv.gz')
  
  set.seed(2342)
  dat <- read_csv(file.path(path,'from_python',run)) %>%
    mutate(preds = if_else(preds == 'LABEL_1',1,0)) %>%
    left_join(read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
                select(id,pseudo=contains(glue('th{th}_fr100'))),
              by='id') %>%
    filter(pseudo != -1)
  
  N <- sum(dat$preds == 1)
  
  dat <- dat %>%
    group_by(preds) %>%
    slice_sample(n=N) %>%
    mutate(tp = if_else((preds == 1) & (pseudo == 1),1,0),
           tn = if_else((preds == 0) & (pseudo == 0),1,0),
           fp = if_else((preds == 1) & (pseudo == 0),1,0),
           fn = if_else((preds == 0) & (pseudo == 1),1,0))
  
  processed <- textProcessor(dat$text,stem=TRUE,metadata=dat,
                             wordLengths=c(2,20),verbose=TRUE)
  
  docs <- prepDocuments(processed$documents,
                        processed$vocab,
                        processed$meta,
                        lower.thresh=25,
                        upper.thresh=round(length(processed$documents)*0.8))
  
  tm <- stm(documents=docs$documents,
            vocab=docs$vocab,
            K=K,verbose=TRUE,init.type='Spectral',seed=123)
  
  write_rds(list(tm=tm,processed=processed,docs=docs),
            file.path(path,'data_out',glue('stm_{th}.rds')))
  
}





