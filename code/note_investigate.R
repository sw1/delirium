mm80 <- c(7008,
          13586,
          25942,
          37953,
          41812,
          51715,
          65108,
          68247,
          90786,
          91536,
          112871,
          141529,
          147027,
          155696,
          159635,
          169068,
          191080,
          202616,
          204218,
          207521,
          232316,
          232885,
          233916,
          244927,
          253087,
          253623,
          254598,
          255246,
          268214,
          276210,
          295280,
          311151,
          312343,
          315244,
          320773,
          326162,
          327772,
          329561,
          333317,
          344480,
          351052,
          380512,
          402093,
          404641,
          405835,
          408543,
          409184,
          421538,
          462866,
          473650,
          495153,
          505336,
          517814,
          517890,
          529290,
          533438,
          541008,
          549062,
          552104,
          571093,
          583000,
          603105,
          605085,
          605518,
          606385,
          606634,
          606958,
          608602,
          609476,
          611697,
          611765,
          614309,
          620730,
          627884,
          628415,
          630559,
          631188,
          632017,
          636425,
          637068,
          639465,
          641015,
          641152,
          642453,
          642722,
          643609,
          644886,
          645352,
          646692,
          650342,
          650430,
          650770,
          654316,
          655378,
          655601,
          657075,
          658419,
          659646,
          660166,
          663070,
          663617,
          666025,
          668124,
          669125,
          669361,
          670088,
          670429,
          671293,
          673074,
          673989,
          674371,
          676349,
          677286,
          677749,
          679018,
          683731)

mm90 <- c(13586,
          30787,
          51715,
          68247,
          81587,
          90786,
          112871,
          138572,
          141529,
          147027,
          152983,
          191080,
          202616,
          204218,
          207521,
          232885,
          233916,
          244927,
          253087,
          268214,
          269661,
          276210,
          295280,
          299769,
          300908,
          311151,
          320773,
          324556,
          327772,
          329561,
          333317,
          340875,
          344480,
          351052,
          360867,
          380512,
          402093,
          405835,
          408543,
          409184,
          421538,
          425999,
          436550,
          462866,
          473650,
          495153,
          505336,
          511047,
          529290,
          533438,
          541008,
          549062,
          552104,
          571093,
          583000,
          603105,
          605085,
          605518,
          606385,
          606634,
          606958,
          608602,
          611697,
          614309,
          620327,
          620730,
          622802,
          627654,
          628669,
          628760,
          630559,
          631188,
          632017,
          636425,
          636746,
          637068,
          639465,
          641015,
          641152,
          642453,
          642722,
          643609,
          644886,
          645352,
          646692,
          646857,
          650342,
          650430,
          652249,
          655378,
          655601,
          657075,
          658419,
          659646,
          660166,
          663070,
          663617,
          666025,
          668124,
          669125,
          669361,
          670088,
          670429,
          671255,
          671293,
          673074,
          673989,
          674371,
          676349,
          677286,
          677609,
          677749,
          679018,
          681741,
          682592,
          683731)

mm90_pl2 <- c(7480,
               13586,
               51715,
               68247,
               81587,
               90786,
               141529,
               163369,
               190471,
               191080,
               202616,
               207521,
               253087,
               255246,
               268214,
               269661,
               276210,
               284648,
               295280,
               299769,
               300908,
               311151,
               320773,
               323570,
               329561,
               333317,
               340875,
               344480,
               351052,
               380512,
               394463,
               402093,
               404641,
               405835,
               409184,
               416605,
               421538,
               425999,
               436550,
               473650,
               499081,
               505336,
               511047,
               529290,
               533438,
               549062,
               552104,
               564475,
               570012,
               571093,
               583000,
               590532,
               603105,
               605085,
               605518,
               606385,
               606634,
               606958,
               608602,
               611047,
               614309,
               620327,
               620730,
               627654,
               628669,
               628760,
               630559,
               631188,
               632017,
               634265,
               636425,
               636746,
               637068,
               639465,
               641015,
               641152,
               642722,
               643609,
               644886,
               645352,
               646692,
               646857,
               650342,
               650430,
               655378,
               655601,
               657075,
               658419,
               659646,
               660166,
               663617,
               666025,
               668124,
               669125,
               670088,
               670429,
               671255,
               673074,
               673989,
               674371,
               676349,
               677286,
               677609,
               677749,
               679018,
               682592,
               683731)

print_note <- function(s, w = 100) {
  n <- ceiling(str_length(s) / w)
  
  chunks <- str_sub(s, 
                    seq(1, by = w, length.out = n), 
                    seq(w, by = w, length.out = n))
  
  for (chunk in chunks) {
    cat(chunk, "\n")
  }
}

icd_lookup <- tibble(icd9cm_hierarchy) %>% 
  mutate_all(str_to_lower) %>%
  select(code,long_desc) %>%
  full_join(tibble(icd10cm2016) %>%  
              mutate_all(str_to_lower) %>%
              select(code,long_desc),
            by='code') %>%
  mutate(long_desc.x = if_else(is.na(long_desc.x),'',long_desc.x),
         long_desc.y = if_else(is.na(long_desc.y),'',long_desc.y)) %>%
  unite('desc',long_desc.x:long_desc.y,sep='; ',remove=TRUE) %>%
  distinct() %>%
  mutate(code=glue('icd_{code}')) %>%
  rename(name=code)


feats <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) 

mm_dis_8090 <- c(setdiff(mm80,mm90),setdiff(mm90,mm80))
mm_dis_pl12 <- c(setdiff(mm90,mm90_pl2),setdiff(mm90_pl2,mm90))

tbl90 <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  filter(id %in% mm90)

tbl_disagree <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  filter(id %in% mm_dis_8090) %>%
  mutate(label_80correct = if_else(!(id %in% mm80),1,0),
         label_90correct = if_else(!(id %in% mm90),1,0)) %>%
  select(id,label,label_90correct,label_80correct,hpi_hc)

# 80 v 90 dictates type 1 v 2 error

idx <- 1 # sbo, label 0 but coded pos delirium
idx <- 2 # confusion, coded pos, label 1?
idx <- 3 # nafld, intubated, depression, stroke, huge risk but 0?, coded for disorientation and restraints
idx <- 4 # presented altered, schizo, tox enceph +/- wernicke, label 0
idx <- 5 # coded delirium, intubated, mention agitated, label 1
idx <- 6 # stroke, label 0
idx <- 7 # toxic enceph, label 0
idx <- 8 # intubated, restraints, enceph, confused, label 0
idx <- 9 # coded delirium, intubated, brain aneurysm, label 0?
idx <- 10 # coded delirium, brain hemorrhage alc
print_note(tbl_disagree$hpi_hc[idx])
feats %>% 
  filter(id == tbl_disagree$id[idx]) %>%
  select(-id,-label,-set) %>% 
  pivot_longer(everything()) %>% 
  filter(value > 0) %>% 
  arrange(desc(value)) %>%
  left_join(icd_lookup,by='name') %>%
  print(n=Inf)
tbl_disagree$id[idx]


feats_fn <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
  select(-set,-label) %>%
  right_join(tbl90 %>% filter(label == 1),by='id') %>%
  select(-id,-set,-starts_with('label'),-hpi,-hc,-hpi_hc) %>%
  summarise_all(mean) %>%
  pivot_longer(everything()) %>%
  arrange(desc(value)) %>%
  filter(value > 0.0) %>%
  rename(value_fn=value) 

feats_fp <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
  select(-set,-label) %>%
  right_join(tbl90 %>% filter(label == 0),by='id') %>%
  select(-id,-set,-starts_with('label'),-hpi,-hc,-hpi_hc) %>%
  summarise_all(mean) %>%
  pivot_longer(everything()) %>%
  arrange(desc(value)) %>%
  filter(value > 0.0) %>%
  rename(value_fp=value) 

feats_tp <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
  filter(label == 1) %>%
  select(-set,-starts_with('label')) %>%
  anti_join(tbl90,by='id') %>%
  select(-id) %>%
  summarise_all(mean) %>%
  pivot_longer(everything()) %>%
  arrange(desc(value)) %>%
  filter(value > 0.0) %>%
  rename(value_tp=value) 

feats_tn <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
  filter(label == 0) %>%
  select(-set,-starts_with('label')) %>%
  anti_join(tbl90,by='id') %>%
  select(-id) %>%
  summarise_all(mean) %>%
  pivot_longer(everything()) %>%
  arrange(desc(value)) %>%
  filter(value > 0.0) %>%
  rename(value_tn=value) 

feats_fn %>%
  full_join(feats_fp) %>%
  full_join(feats_tp) %>%
  full_join(feats_tn) %>%
  filter(value_fn > 0.2 | value_fp > 0.2 | value_tp > 0.2 | value_tn > 0.2) %>%
  select(name,value_tp,value_fn,value_tn,value_fp) %>%
  print(n=Inf)


tbl_disagree$hpi_hc[1]





tbl_inc <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  filter(id %in% mm)

tbl1 <- tbl_inc %>%
  filter(label == 1)

tbl0 <- tbl_inc %>%
  filter(label == 0)

feats1 <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
  filter(id %in% tbl1$id)

feats0 <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
  filter(id %in% tbl0$id)

feats_correct <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
  filter(!(id %in% tbl_inc$id))

feats10 <- feats1 %>% 
  select(-id,-set,-label) %>%
  summarise_all(mean) %>%
  pivot_longer(everything()) %>%
  arrange(desc(value)) %>%
  filter(value > 0.0) %>%
  rename(value_fn=value) 

feats01 <- feats0 %>% 
  select(-id,-set,-label) %>%
  summarise_all(mean) %>%
  pivot_longer(everything()) %>%
  arrange(desc(value)) %>%
  filter(value > 0.0) %>%
  rename(value_fp=value) 

feats11 <- feats_correct %>% 
  filter(label == 1) %>%
  select(-id,-set,-label) %>%
  summarise_all(mean) %>%
  pivot_longer(everything()) %>%
  arrange(desc(value)) %>%
  filter(value > 0.0) %>%
  rename(value_tp=value) 

feats00 <- feats_correct %>% 
  filter(label == 0) %>%
  select(-id,-set,-label) %>%
  summarise_all(mean) %>%
  pivot_longer(everything()) %>%
  arrange(desc(value)) %>%
  filter(value > 0.0) %>%
  rename(value_tn=value) 
  
feats10 %>%
  full_join(feats01) %>%
  full_join(feats11) %>%
  full_join(feats00) %>%
  filter(value_fn > 0.2 | value_fp > 0.2 | value_tp > 0.2 | value_tn > 0.2) %>%
  select(name,value_tp,value_fn,value_tn,value_fp) %>%
  print(n=Inf)


fp_deldd <- tbl_inc %>%
  filter(label == 0) %>%
  select(id,hpi_hc) %>%
  left_join(feats0,by='id') %>%
  filter(count_del_dd > 0)

print_note(fp_deldd$hpi_hc[1])






mm_plf <- str_replace(list.files(file.path(path,'from_python','p90'),
                                 pattern='.csv.gz'),
                      '.csv.gz','') %>%
  as.integer()

mm_plam <- str_replace(list.files(file.path(path,'from_python','pllama'),
                                 pattern='.csv.gz'),
                      '.csv.gz','') %>%
  as.integer()




tbl <- read_csv(file.path(path,'to_python','tbl.csv.gz'))
feats <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds'))

tbl %>%
  select(id,label) %>%
  filter(label != -1) %>%
  left_join(feats %>% select(-label,-set),by='id') %>%
  mutate(lf_mm=if_else(id %in% mm_plf,1,0),
         llama_mm=if_else(id %in% mm_plam,1,0),
         pred_llama=case_when(
           label == 1 & llama_mm == 1 ~ 'llama_fn',
           label == 0 & llama_mm == 1 ~ 'llama_fp',
           label == 1 & llama_mm == 0 ~ 'llama_tp',
           label == 0 & llama_mm == 0 ~ 'llama_tn'
         ),
         pred_lf=case_when(
           label == 0 & lf_mm == 1 ~ 'lf_fp',
           label == 1 & lf_mm == 0 ~ 'lf_tp',
           label == 0 & lf_mm == 0 ~ 'lf_tn',
           label == 1 & lf_mm == 1 ~ 'lf_fn'
         )) %>%
  select(-ends_with('_mm')) %>%
  pivot_longer(starts_with('pred'),names_to='model',values_to='pred') %>%
  mutate(model=str_replace(model,'pred_',''),
         pred=str_replace(pred,'^.*_','')) %>%
  select(-id,-label) %>%
  group_by(model,pred) %>%
  reframe(across(everything(),\(x) mean(x, na.rm = TRUE))) %>%
  pivot_longer(-c(model,pred),names_to='feature',values_to='mean') %>%
  pivot_wider(names_from=c(model,pred),values_from=mean) %>%
  rowwise() %>%
  mutate(d_fn=abs(lf_fn - llama_fn),
         d_fp=abs(lf_fp - llama_fp),
         d_tp=abs(lf_tp - llama_tp),
         d_tn=abs(lf_tn - llama_tn),
         d_max=max(c_across(starts_with('d_')))) %>%
  ungroup() %>%
  arrange(desc(d_max)) %>%
  select(-starts_with('d_')) %>%
  select(feature,
         ends_with('_tp'),
         ends_with('_tn'),
         ends_with('_fp'),
         ends_with('_fn')) %>%
  print(n=Inf)





tbl <- read_csv(file.path(path,'to_python','tbl.csv.gz'))
feats <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds'))

icd9_lookup <- tibble(icd9cm_hierarchy) %>% mutate_all(str_to_lower)
icd10_lookup <- tibble(icd10cm2016) %>%  mutate_all(str_to_lower)

runs <- list.files(file.path(path,'from_python'),pattern='^fkw')
preds <- tibble()
for (run in runs){
  params <- str_match_all(
    run,'th(\\d+).*pl(\\d+).*labpseudo_([a-z]+)[[:punct:]](.*)')[[1]][,-1] %>%
    tibble() %>%
    rename(value=1) %>%
    mutate(p=c('th','pl','lab','mod')) %>%
    pivot_wider(names_from='p',values_from='value') %>%
    mutate(mod=if_else(str_detect(mod,'llama'),'lam','lf'))
    
  preds <- preds %>%
    bind_rows(read_csv(file.path(path,'from_python',run)) %>%
      group_by(id) %>%
      filter(i == max(i)) %>%
      ungroup() %>%
      select(id,labels,preds) %>%
      bind_cols(params))
}

feats <- preds %>%
  mutate(preds_type=case_when(
    labels == 1 & preds == 1 ~ 'tp',
    labels == 0 & preds == 1 ~ 'fp',
    labels == 1 & preds == 0 ~ 'fn',
    labels == 0 & preds == 0 ~ 'tn'
  )) %>%
  left_join(read_rds(
    file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
      select(-set,-label),
    by='id') %>%
  select(-id) %>%
  mutate(across(where(is.numeric), ~ rescale(.x,to=c(0,10)))) %>%
  group_by(th,pl,lab,mod,preds_type) %>%
  summarise_all(mean)  %>%
  pivot_longer(cols=-c(th, pl, lab, mod, preds_type, labels, preds),
               names_to='feature',values_to='mean') %>%
  ungroup()

feats %>%
  filter(mod == 'lf',
         pl == '1',
         lab == 'expert',
         preds_type %in% c('fn','fp')) %>%
  select(th,feature,preds_type,mean) %>%
  group_by(feature,preds_type) %>%
  mutate(diff=diff(mean)) %>%
  group_by(th,preds_type) %>%
  arrange(desc(abs(diff))) %>%
  group_by(preds_type) %>%
  mutate(rank=dense_rank(desc(abs(diff)))) %>%
  ungroup() %>%
  filter(rank <= 10) %>%
  mutate(feature=factor(feature,levels=unique(feature),ordered=TRUE),
         preds_type = if_else(preds_type == 'fn',
                              'False Negatives',
                              'False Positives')) %>%
  left_join(icd9_lookup %>% 
    select(feature=code,short_desc) %>%
    bind_rows(icd10_lookup %>% select(feature=code,short_desc)) %>%
    distinct() %>%
    mutate(feature = paste0('icd_',feature)),
    by='feature') %>%
  mutate(feature=case_when(
    !is.na(short_desc) ~ short_desc,
    str_detect(feature,'count_service') ~ glue('service: {str_match(feature,"service_(.*)")[,2]}'),
    feature == 'count_home' ~ 'discharge: home',
    feature == 'count_geri' ~ 'hospital_course: geriatr',
    TRUE ~ feature)) %>%
  ggplot(aes(x=feature,y=mean,group=th,fill=th)) +
  geom_col(color='black',width=.7,position='dodge',alpha=.2) +
  facet_wrap(~preds_type, scales='free_y',ncol=1) +
  coord_flip() +
  theme_classic() +
  theme(legend.position = 'bottom') +
  labs(x='',y='Scaled Mean Occurance',
       fill='Pseudo-Label Threshold') +
  scale_fill_brewer(type='qual',palette='Set1') 


feats %>%
  filter(pl == '1',
         th == '90',
         lab == 'expert',
         preds_type %in% c('fn','fp')) %>%
  select(mod,feature,preds_type,mean) %>%
  group_by(feature,preds_type) %>%
  mutate(diff=diff(mean)) %>%
  group_by(mod,preds_type) %>%
  arrange(desc(abs(diff))) %>%
  group_by(preds_type) %>%
  mutate(rank=dense_rank(desc(abs(diff)))) %>%
  ungroup() %>%
  filter(rank <= 10) %>%
  mutate(feature=factor(feature,levels=unique(feature),ordered=TRUE),
         mod = if_else(mod == 'lf','Longformer','Llama'),
         preds_type = if_else(preds_type == 'fn',
                              'False Negatives',
                              'False Positives')) %>%
  left_join(icd9_lookup %>% 
              select(feature=code,short_desc) %>%
              bind_rows(icd10_lookup %>% select(feature=code,short_desc)) %>%
              distinct() %>%
              mutate(feature = paste0('icd_',feature)),
            by='feature') %>%
  mutate(feature=case_when(
    !is.na(short_desc) ~ short_desc,
    str_detect(feature,'count_service') ~ glue('service: {str_match(feature,"service_(.*)")[,2]}'),
    feature == 'count_home' ~ 'discharge: home',
    feature == 'count_geri' ~ 'hospital_course: geriatr',
    feature == 'count_del_dd' ~ 'discharge dx: delirium',
    TRUE ~ feature)) %>%
  ggplot(aes(x=feature,y=mean,group=mod,fill=mod)) +
  geom_col(color='black',width=.7,position='dodge',alpha=.2) +
  facet_wrap(~preds_type, scales='free_y',ncol=1) +
  coord_flip() +
  theme_classic() +
  theme(legend.position = 'bottom') +
  labs(x='',y='Scaled Mean Occurance',
       fill='Model') +
  scale_fill_brewer(type='qual',palette='Set1') 

feats %>%
  filter(mod == 'lf',
         th == '90',
         lab == 'expert',
         preds_type %in% c('fn','fp')) %>%
  select(pl,feature,preds_type,mean) %>%
  group_by(feature,preds_type) %>%
  mutate(diff=diff(range(mean))) %>%
  group_by(pl,preds_type) %>%
  arrange(desc(abs(diff))) %>%
  group_by(preds_type) %>%
  mutate(rank=dense_rank(desc(abs(diff)))) %>%
  ungroup() %>%
  filter(rank <= 10, pl != 3) %>%
  mutate(feature=factor(feature,levels=unique(feature),ordered=TRUE),
         pl = case_when(
           pl == 1 ~ 'FT',
           pl == 2 ~ 'FT + PT',
           pl == 3 ~ 'FT + PT + T',
         ),
         preds_type = if_else(preds_type == 'fn',
                              'False Negatives',
                              'False Positives')) %>%
  left_join(icd9_lookup %>% 
              select(feature=code,short_desc) %>%
              bind_rows(icd10_lookup %>% select(feature=code,short_desc)) %>%
              distinct() %>%
              mutate(feature = paste0('icd_',feature)),
            by='feature') %>%
  mutate(feature=case_when(
    !is.na(short_desc) ~ short_desc,
    str_detect(feature,'count_service') ~ glue('service: {str_match(feature,"service_(.*)")[,2]}'),
    feature == 'count_home' ~ 'discharge: home',
    feature == 'count_geri' ~ 'hospital_course: geriatr',
    feature == 'count_del_dd' ~ 'discharge dx: delirium',
    TRUE ~ feature)) %>%
  ggplot(aes(x=feature,y=mean,group=pl,fill=pl)) +
  geom_col(color='black',width=.7,position='dodge',alpha=.2) +
  facet_wrap(~preds_type, scales='free_y',ncol=1) +
  coord_flip() +
  theme_classic() +
  theme(legend.position = 'bottom') +
  labs(x='',y='Scaled Mean Occurance',
       fill='Pipeline') +
  scale_fill_brewer(type='qual',palette='Set1') 

preds %>%
  select(-id) %>%
  filter(lab == 'unlabeled') %>%
  group_by(th,pl,mod) %>%
  summarise(pos=sum(preds),
            neg=sum(preds == 0))

preds %>%
  filter(lab == 'unlabeled',
         !(mod == 'lf' & pl == 3),
         pl != 2) %>%
  select(-lab,-labels) %>%
  unite('mod',th:mod,sep='_') %>%
  pivot_wider(names_from='mod',values_from='preds') %>%
  rowwise() %>%
  filter(sum(c_across(-id)) > 0) %>%
  ungroup() %>%
  left_join(read_csv(file.path(path,'to_python','tbl.csv.gz')) %>% 
              select(id,hpi_hc),
            by='id')
