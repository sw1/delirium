tbl <- read_csv(file.path(path,'to_python','tbl.csv.gz'))
feats <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds'))
master <- read_rds(file.path(path,'data_out','03_tbl_final.rds'))

tbl %>%
  filter(set == 'train',label == -1) %>%
  select(matches('fr100'),label) %>%
  select(matches('pseudo'),'label') %>%
  filter(if_all(everything(), ~ .x == 1))

tbl2 <- tbl %>%
  filter(set == 'train',
         label == -1) %>%
  select(id,contains('fr100')) %>%
  select(id,contains('pseudo')) %>%
  filter(if_all(-id, ~ .x == 1)) %>%
  left_join(feats,by='id') %>%
  left_join(master %>% select(id,discharge_disposition),by='id')

tbl3 <- tbl %>%
  filter(set == 'train',
         label == -1) %>%
  select(id,contains('fr100')) %>%
  select(id,contains('pseudo')) %>%
  filter(label_pseudo_th80_fr100 != label_pseudo_th90_fr100,
         label_pseudo_th80_fr100 != -1,
         label_pseudo_th90_fr100 != -1) %>%
  select(id) %>%
  left_join(feats,by='id') %>%
  left_join(master %>% select(id,discharge_disposition),by='id')
  

tbl2 %>%
  mutate(dd_ext=as.integer(str_count(discharge_disposition,'extended|servic')>0),
         dd_home=as.integer(str_count(discharge_disposition,'home')>0)) %>%
  reframe(sum(count_del > 0)/n(),
          mean(los),
          sd(los),
          mean(age),
          mean(count_ciwa > 0),
          mean(count_psych_med > 0),
          sd(age),
          sum(dd_home,na.rm = TRUE)/n(),
          sum(dd_ext,na.rm = TRUE)/n()) %>%
  pivot_longer(everything())
 

tbl3 %>%
  mutate(dd_ext=as.integer(str_count(discharge_disposition,'extended|servic')>0),
         dd_home=as.integer(str_count(discharge_disposition,'home')>0)) %>%
  reframe(sum(count_del > 0)/n(),
          mean(los),
          sd(los),
          mean(age),
          mean(count_ciwa > 0),
          mean(count_psych_med > 0),
          sd(age),
          sum(dd_home,na.rm = TRUE)/n(),
          sum(dd_ext,na.rm = TRUE)/n()) %>%
  pivot_longer(everything())
