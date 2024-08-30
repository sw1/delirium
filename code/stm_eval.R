pacman::p_load(tidyverse,glue,gtsummary,flextable,icd.data,tidymodels,
               rpart,rpart.plot,officer,gridExtra,stm,LDAvis)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))



weights <- read_csv(file.path(path,'data_out','stm_weights.csv.gz')) %>%
  mutate(feature=case_when(
    feature == 'preds' & w > 0 ~ 'pred_pos',
    feature == 'preds' & w < 0 ~ 'pred_neg',
    TRUE ~ feature
  ),w=if_else(feature == 'pred_neg',abs(w),w),
  th = as.character(th)) %>%
  group_by(feature,th) %>%
  mutate(rank=dense_rank(desc(w))) %>%
  ungroup() %>%
  filter(rank <= 5)


terms <- tibble()
ths <- c('70','80','90')
for (th in ths){
  
  tm <- read_rds(file.path(path,'data_out',glue('stm_{th}.rds')))$tm

  top_words <- labelTopics(tm,n=20)

  top_freq <- as_tibble(top_words$prob) %>%
    mutate(K=row_number(),
           stat='freq',
           th=th)
  
  top_frex <- as_tibble(top_words$frex) %>%
    mutate(K=row_number(),
           stat='frex',
           th=th)

  terms <- terms %>%
    bind_rows(top_freq) %>%
    bind_rows(top_frex)
  
}

top_freqs <- weights %>%
  left_join(terms %>% filter(stat == 'freq'),by=c('th','K')) 

top_frex <- weights %>%
  left_join(terms %>% filter(stat == 'frex'),by=c('th','K')) 

top_terms <- bind_rows(top_freqs,top_frex) %>%
  select(-w) %>%
  arrange(th,feature,rank)

write_csv(top_terms,file.path(path,'data_out','stm_top_terms.csv.gz'))


# top_terms %>%
#   filter(th == '90',feature %in% c('tp','tn')) %>%
#   View()
# dat <- read_rds(file.path(path,'data_out','stm_90.rds'))
# tm <- dat$tm
# docs <- dat$docs
# 
# toLDAvis(tm,docs=docs$documents)




