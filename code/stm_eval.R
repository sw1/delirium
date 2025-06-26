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
  th = as.character(th),
  pl = as.character(pl)) %>%
  group_by(feature,th,pl,mod) %>%
  mutate(rank=dense_rank(desc(w))) %>%
  ungroup() %>%
  filter(rank <= 5)

params <- weights %>%
  select(th,pl,mod) %>%
  distinct() 

terms <- tibble()
for (i in 1:nrow(params)){
  
  th <- params$th[i]
  pl <- params$pl[i]
  mod <- params$mod[i]
  
  tm <- read_rds(file.path(path,'data_out',
                           glue('stm_th{th}_pl{pl}_mod{mod}.rds')))$tm

  top_words <- labelTopics(tm,n=20)

  top_freq <- as_tibble(top_words$prob) %>%
    mutate(K=row_number(),
           stat='freq',
           pl=pl,
           mod=mod,
           th=th)
  
  top_frex <- as_tibble(top_words$frex) %>%
    mutate(K=row_number(),
           stat='frex',
           pl=pl,
           mod=mod,
           th=th)

  terms <- terms %>%
    bind_rows(top_freq) %>%
    bind_rows(top_frex)
  
}

top_freqs <- weights %>%
  left_join(terms %>% filter(stat == 'freq'),
            by=c('th','K','pl','mod')) 

top_frex <- weights %>%
  left_join(terms %>% filter(stat == 'frex'),
            by=c('th','K','pl','mod')) 

top_terms <- bind_rows(top_freqs,top_frex) %>%
  select(mod,pl,th,feature,K,rank,stat,starts_with('V')) %>%
  arrange(desc(th),pl,desc(mod),feature,rank)

write_csv(top_terms,file.path(path,'data_out','stm_top_terms.csv.gz'))


# top_terms %>%
#   filter(th == '90',
#          pl == '1',
#          feature %in% c('tp','tn')) %>%
#   arrange(feature,rank,mod) %>%
#   View()
# dat <- read_rds(file.path(path,'data_out','stm_90.rds'))
# tm <- dat$tm
# docs <- dat$docs
# 
# toLDAvis(tm,docs=docs$documents)




