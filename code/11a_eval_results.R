pacman::p_load(tidyverse,glue,Rtsne,icd.data)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

fs <- read_rds(file.path(path,'data_in','08_rf_fs.rds')) 

# select best model with reduced features 
# within 1 sd of best performing model based on rmse
params <- fs$perf %>%
  filter(rmse < min(rmse) + sd(rmse),
         n_feats < 200) %>%
  arrange(rmse) %>%
  slice_head(n=1)

features_subset <- fs$features %>%
  arrange(desc(Importance)) %>%
  slice_head(n=params$n_feats[1]) %>%
  pull(Variable) %>%
  paste('^',.,'$',sep='')

# generate importance table
icd9_lookup <- tibble(icd9cm_hierarchy) %>% mutate_all(str_to_lower)
icd10_lookup <- tibble(icd10cm2016) %>%  mutate_all(str_to_lower)

# load dataset and filter features, nrows 154267
master <- read_rds(
  file.path(path,'data_in','09_alldat_preprocessed_for_pred.rds')) %>%
  select(id,set,label,matches(features_subset))

fn <- 'tbl_to_python_expertupdate_chunked_rfst_majvote_th90_ns0.csv.gz'
tbl <- read_csv(file.path(path,'to_python',fn))

tbl %>% 
  filter(label_pseudo != -1,label_icd != -1) %>% 
  reframe(n(),mean(label_pseudo == label_icd)) 

disagree <- tbl %>% 
  filter(label_pseudo != -1,label_icd != -1) %>%
  filter(label_pseudo != label_icd) %>%
  arrange(label_icd) %>%
  select(id,hpi,hc,label_pseudo,label_icd) %>%
  distinct() %>%
  mutate(hpi_hc = glue('{hpi} {hc}')) %>%
  select(-hpi,-hc)

fn <- 'tbl_to_python_expertupdate_chunked_rfst_majvote_th70_ns0.csv.gz'
tbl <- read_csv(file.path(path,'to_python',fn))

master %>% 
  select(-label,-set) %>%
  left_join(tbl %>% select(id,set,label,label_pseudo),by='id') %>%
  mutate(year=year(as_date(discharge_date))) %>%
  filter(str_detect(set,'expert')) %>%
  group_by(label,year) %>%
  reframe(n=n()) %>%
  group_by(year) %>%
  mutate(mean=n/sum(n)) %>%
  ggplot(aes(year,mean,fill=as.factor(label))) +
  geom_bar(stat='identity',position='stack') 

master %>% 
  select(-label,-set) %>%
  left_join(tbl %>% select(id,set,label,label_pseudo),by='id') %>%
  filter(str_detect(set,'expert')) %>%
  group_by(label,age) %>%
  mutate(age=ntile(age,16)) %>%
  group_by(label,age) %>%
  reframe(n=n()) %>%
  group_by(age) %>%
  mutate(mean=n/sum(n)) %>%
  ggplot(aes(age,mean,fill=as.factor(label))) +
  geom_bar(stat='identity',position='stack') 

master %>% 
  select(-label,-set) %>%
  left_join(tbl %>% select(id,set,label,label_pseudo),by='id') %>%
  filter(str_detect(set,'expert')) %>%
  ggplot(aes(x=count_del,fill=as.factor(label))) +
  geom_density(alpha=.3,adjust=5) 

master %>% 
  select(-label,-set) %>%
  left_join(tbl %>% select(id,set,label,label_pseudo),by='id') %>%
  mutate(year=year(as_date(discharge_date))) %>%
  filter(str_detect(set,'expert')) %>%
  ggplot(aes(x=count_del,fill=as.factor(label))) +
  geom_density(alpha=.3,adjust=5) +
  facet_wrap(~year)
    
master %>% 
  select(-label,-set) %>%
  left_join(tbl %>% select(id,set,label,label_pseudo),by='id') %>%
  mutate(year=year(as_date(discharge_date))) %>%
  filter(str_detect(set,'expert')) %>%
  ggplot(aes(year,los,color=as.factor(label))) +
  stat_smooth(method='loess',se=FALSE)

master %>% 
  select(-label,-set) %>%
  left_join(tbl %>% select(id,set,label,label_pseudo),by='id') %>%
  mutate(year=year(as_date(discharge_date))) %>%
  filter(str_detect(set,'expert')) %>%
  ggplot(aes(year,len_pmhx,color=as.factor(label))) +
  stat_smooth(method='loess',se=FALSE)

tbl %>%
  select(id,hpi,hc,label_pseudo,label_icd) %>%
  distinct() %>%
  mutate(hpi_hc = glue('{hpi} {hc}')) %>%
  select(-hpi,-hc) %>%
  filter(label_pseudo != -1,
         label_icd != -1,
         label_pseudo != label_icd)

master %>% 
  filter(id == 31) %>%
  select(-id,-set,-label) %>%
  pivot_longer(everything(),cols=,names_to='Feature',values_to='Value') %>%
  mutate(code=str_replace(Feature,'^icd_','')) %>%
  left_join(icd9_lookup  %>% select(code,long_desc),
            by='code') %>%
  left_join(icd10_lookup %>% select(code,long_desc),
            by='code') %>%
  filter(Value > 0) %>%
  mutate(Feature=case_when(
    !is.na(long_desc.x) ~ long_desc.x,
    !is.na(long_desc.y) ~ long_desc.y,
    !is.na(code) ~ code,
    TRUE ~ NA
  )) %>%
  select(Feature,Value) %>%
  print(n=200) 

master %>% 
  filter(id == 249) %>%
  select(-id,-set,-label) %>%
  pivot_longer(everything(),cols=,names_to='Feature',values_to='Value') %>%
  mutate(code=str_replace(Feature,'^icd_','')) %>%
  left_join(icd9_lookup  %>% select(code,long_desc),
            by='code') %>%
  left_join(icd10_lookup %>% select(code,long_desc),
            by='code') %>%
  filter(Value > 0) %>%
  mutate(Feature=case_when(
    !is.na(long_desc.x) ~ long_desc.x,
    !is.na(long_desc.y) ~ long_desc.y,
    !is.na(code) ~ code,
    TRUE ~ NA
  )) %>%
  select(Feature,Value) %>%
  print(n=200) 

tbl %>%
  mutate(hpi_hc = glue('{hpi} {hc}')) %>%
  filter(id == 249) %>%
  select(hpi_hc) %>%
  distinct() %>%
  pull(hpi_hc)


ids <- tbl %>%
  select(id,hpi,hc,label_pseudo,label_icd) %>%
  distinct() %>%
  mutate(hpi_hc = glue('{hpi} {hc}')) %>%
  select(-hpi,-hc) %>%
  filter(label_pseudo != -1,
         label_icd != -1,
         label_pseudo != label_icd) %>%
  pull(id)

x <- master %>% 
  filter(id %in% ids) 
x_ids <- x %>% pull(id)
x <- x %>%
  select(-id,-set,-label) %>%
  select(where(~sum(.x) > 0)) 
  # mutate(across(everything(),z))


tsne <- Rtsne(x,normalize=TRUE,pca_center=TRUE,pca_scale=TRUE)

as_tibble(tsne$Y) %>%
  bind_cols(id=x_ids) %>%
  left_join(tbl %>% select(id,label_pseudo,label_icd),by='id') %>%
  ggplot(aes(x=V1,y=V2,color=as.factor(disagree))) +
  geom_point() +
  theme_minimal() +
  theme(legend.position='none') +
  labs(title='t-SNE of Disagreements') 


ids <- tbl %>%
  select(id,hpi,set,hc,label_pseudo,label) %>%
  distinct() %>%
  mutate(hpi_hc = glue('{hpi} {hc}')) %>%
  select(-hpi,-hc) %>%
  filter(label_pseudo != -1 | 
           (set == 'test_expert' & label != -1) | 
           (set == 'train_expert' & label != -1)) %>%
  mutate(grp = case_when(
    set == 'test_expert' & label != -1 ~ 'testexp',
    set == 'train_expert' & label != -1 ~ 'trainexp',
    !(set %in% c('test_expert','train_expert')) & label_pseudo != -1 ~ 'pseudo',
    TRUE ~ 'other'
  )) %>%
  filter(grp != 'other') %>%
  select(id,grp,label,label_pseudo)

x <- ids %>%
  left_join(master %>% select(-set,-label),by='id') %>%
  group_by(grp) %>%
  mutate(n=n()) %>%
  ungroup() %>%
  mutate(n=if_else(min(n) < 1500,min(n),1500)) %>%
  group_by(grp) %>%
  sample_n(unique(n)) %>%
  ungroup() %>%
  select(-n)

x_ids <- x %>% pull(id)
x <- x %>%
  select(-id,-grp,-label) %>%
  select(where(~sum(.x) > 0)) 

tsne2 <- Rtsne(x,normalize=TRUE,pca_center=TRUE,pca_scale=TRUE,
               check_duplicates=FALSE)

as_tibble(tsne2$Y) %>%
  bind_cols(tibble(id=x_ids) %>% left_join(ids,by='id')) %>%
  mutate(label=if_else(label == 1,
                       'label_pos',if_else(label == 0,'label_neg','unk'))) %>%
  ggplot(aes(x=V1,y=V2,color=grp)) +
  geom_point() +
  facet_wrap(as.factor(label)~as.factor(label_pseudo)) +
  theme_minimal() +
  theme(legend.position='none') +
  labs(title='t-SNE of Disagreements') 
