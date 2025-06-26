pacman::p_load(tidyverse,glue,gtsummary,flextable,icd.data,tidymodels,
               rpart,rpart.plot,officer,gridExtra,grid,ggrepel)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

border_style <- officer::fp_border(color='black', width=1)
read_csv(file.path(path,'res','15_lasso_results.csv.gz')) %>%
  filter(!is.na(b_acc),
         fraction == 100 | is.na(fraction)) %>%
  mutate(threshold=if_else(is.na(threshold),'',as.character(threshold)),
         w = as.integer(w),
         lambda = num(lambda, digits=3), 
         across(where(is.double) & !lambda, ~num(.x, digits = 2)),  
         prop1=num(prop1*100,label='%'),
         label=case_when(
           label == 'fullexpert' ~ 'Liberal',
           label == 'onlyexpert' ~ 'Strict',
           label == 'icd' ~ 'ICD',
           label == 'pseudo' ~ 'Pseudo'
         )) %>%
  select(-fn,-fraction) %>%
  mutate(label=factor(label,levels=c('Pseudo','Liberal','Strict','ICD'),
                    ordered=TRUE)) %>%
  arrange(label,desc(threshold),lambda,w,desc(b_acc)) %>%
  select('Label'=label,
         'Thres.'=threshold,
         'Lambda'=lambda,
         'Class Wt.'=w,
         'BAcc.'=b_acc,
         'Prec.'=prec,
         'Rec.'=rec,
         'F1'=f1,
         'Pred. 1 (%)'=prop1) %>%
  flextable() %>%
  set_header_labels(Lambda = "L1 λ") %>%
  autofit() %>%
  add_header_row(top=TRUE,
                 values=c('Labeling','','Lasso','','Testing','','','','')) %>%
  merge_at(i=1,j=1:2,part='header') %>%
  merge_at(i=1,j=3:4,part='header') %>%
  merge_at(i=1,j=5:9,part='header') %>%
  vline(part='all',j=2,border=border_style) %>%
  vline(part='all',j=4,border=border_style) %>%
  flextable::align(align='center',j=1:9,part='all') %>%
  fontsize(size=9,part='body') %>%
  fontsize(i=1:2,size=11,part='header') %>%
  save_as_docx(path=file.path(path,'tbls','lasso.docx'))
  

# generate data summary table
tbl1 <- read_rds(file.path(path,'data_out','03_tbl_final.rds')) %>% 
  mutate(sex=if_else(sex==0,'female','male'),
         label=case_when(
           is.na(label) ~ 'Unlabeled',
           label == 1 ~ 'Label: 1',
           label == 0 ~ 'Label: 0',
           TRUE ~ NA),
         num_allergies = round(num_allergies),
         len_pmhx = round(len_pmhx)) %>%
  select(label,service,sex,age,los,num_meds,num_allergies,len_pmhx) %>%
  tbl_summary(by=label,
              statistic=list(all_continuous() ~ '{mean} ({sd})',       
                             all_categorical() ~ '{n} ({p}%)'),   
              digits=all_continuous() ~ 1,                             
              type=all_categorical() ~ 'categorical',                
              label=list(                                           
                label ~ 'Label', 
                age ~ 'Age (Years)',
                sex ~ 'Sex',
                los ~ 'Length of Stay (Days)',
                service ~ 'Service',
                num_meds ~ 'Medications on Admission (Count)',
                num_allergies ~ 'Allergies on Admission (Count)',
                len_pmhx ~ 'Length of Past Medical History (Characters)'),
              missing_text="Missing") %>%
  add_p(age ~ 'kruskal.test') %>%
  as_gt() %>%
  gt::gtsave(file.path(path,'tbls','demo_tbl.docx'))

# 1. read + wrangle your data
df <- read_rds(file.path(path, "data_out", "03_tbl_final.rds")) %>%
  mutate(
    sex = if_else(sex == 0, "female", "male"),
    label = case_when(
      is.na(label)   ~ "Unlabeled",
      label == 1     ~ "Label: 1",
      label == 0     ~ "Label: 0",
      TRUE           ~ NA_character_
    ),
    num_allergies = round(num_allergies),
    len_pmhx       = round(len_pmhx)
  ) %>%
  select(label, service, sex, age, los, num_meds, num_allergies, len_pmhx)

# 2. build your summary table (no add_p)
tbl <- df %>%
  tbl_summary(
    by = label,
    statistic = list(
      all_continuous()  ~ "{mean} ({sd})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = all_continuous() ~ 1,
    type   = all_categorical() ~ "categorical",
    label  = list(
      label         ~ "Label",
      service       ~ "Service",
      sex           ~ "Sex",
      age           ~ "Age (Years)",
      los           ~ "Length of Stay (Days)",
      num_meds      ~ "Medications on Admission (Count)",
      num_allergies ~ "Allergies on Admission (Count)",
      len_pmhx      ~ "Length of Past Medical History (Characters)"
    ),
    missing_text = "Missing"
  ) %>%
  
  # 3. add the MAD column
  modify_table_body(
    ~ .x %>%
      rowwise() %>%
      mutate(
        MAD = {
          # pull out every “stat_…” column (one per label)
          raw_vals <- c_across(starts_with("stat_"))
          # drop everything except the number
          nums     <- parse_number(raw_vals)
          # mean absolute deviation
          round(mean(abs(nums - mean(nums))), 1)
        }
      ) %>%
      ungroup()
  ) %>%
  
  # 4. give it a nice header
  modify_header(
    list(MAD ~ "**MAD**")
  )

# 5. render + save
tbl %>%
  as_gt() %>%
  gt::gtsave(file.path(path, "tbls", "demo_tbl2.docx"))

# generate data summary table
tbl1 <- read_rds(file.path(path,'data_out','03_tbl_final.rds')) %>% 
  left_join(read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
              select(id,set),
            by='id') %>%
  mutate(sex=if_else(sex==0,'female','male'),
         set=case_when(
           set == 'train' ~ 'Train',
           set == 'val' ~ 'Validation',
           set == 'heldout_expert' ~ 'Test'),
         set=factor(set,levels=c('Train','Validation','Test'),ordered=TRUE)) %>%
  select(set,service,sex,age,los,num_meds,num_allergies,len_pmhx) %>%
  tbl_summary(by=set,
              statistic=list(all_continuous() ~ '{mean} ({sd})',       
                             all_categorical() ~ '{n} ({p}%)'),   
              digits=all_continuous() ~ 1,                             
              type=all_categorical() ~ 'categorical',                
              label=list(                                           
                set ~ 'Set', 
                age ~ 'Age (Years)',
                sex ~ 'Gender',
                los ~ 'Length of Stay (Days)',
                service ~ 'Service',
                num_meds ~ 'Medications on Admission (Count)',
                num_allergies ~ 'Allergies on Admission (Count)',
                len_pmhx ~ 'Length of Past Medical History'),
              missing_text="Missing") %>%
  add_p(age ~ 'kruskal.test') %>%
  as_gt() %>%
  gt::gtsave(file.path(path,'tbls','demo_tbl_set.docx'))


# generate rf cv performance tbl
perf <- read_rds(file.path(path,'data_out','08_rf_fs.rds'))

border_style <- officer::fp_border(color='black', width=1)
perf$perf %>%
  arrange(rmse) %>%
  mutate(n_feats=as.integer(n_feats),
         mtry=as.integer(mtry),
         across(where(is.double),~num(.x,digits=2)),
         prop1=num(prop1*100,label='%'),
         min_node_perc=num(min_node_perc*100,label='%')) %>%
  select('Feat. (N)'=n_feats,
         'Mtry'=mtry,
         'N Size (%)'=min_node_perc,
         'RMSE'=rmse,
         'BAcc.'=bacc,
         'Prec.'=prec,
         'Rec.'=rec,
         'F1'=f1,
         'Pred. 1 (%)'=prop1) %>%
  flextable() %>%
  autofit() %>%
  add_header_row(top=TRUE,
                 values=c('','','','RF','Validation','','','','')) %>%
  merge_at(i=1,j=1:3,part='header') %>%
  merge_at(i=1,j=5:9,part='header') %>%
  # merge_at(i=1,j=8:9,part='header') %>%
  vline(part='all',j=3,border=border_style) %>%
  vline(part='all',j=4,border=border_style) %>%
  bg(.,i=~`RMSE` < (min(`RMSE`) + sd(`RMSE`)),
     part='body',bg='lightgray') %>%
  flextable::align(align='center',j=1:9,part='all') %>%
  bold(i=~`BAcc.` == max(`BAcc.`),j=5,bold=TRUE,part='body') %>%
  bold(i=~`Prec.` == max(`Prec.`),j=6,bold=TRUE,part='body') %>%
  bold(i=~`Rec.` == max(`Rec.`),j=7,bold=TRUE,part='body') %>%
  bold(i=~F1 == max(F1),j=8,bold=TRUE,part='body') %>%
  bold(i=~`RMSE` == min(`RMSE`),j=4,bold=TRUE,part='body') %>%
  bold(i=~`Feat. (N)` < 200,j=1,bold=TRUE,part='body') %>%
  bg(.,i=~`Feat. (N)` < 200 & `RMSE` < 0.2,
     part='body',bg='gray') %>%
  fontsize(size=9, part='body') %>%
  fontsize(i=1:2,size=11,part='header') %>%
  save_as_docx(path=file.path(path,'tbls','rf_cv.docx'))

params <- read_rds(file.path(path,'data_out','08_rf_fs.rds'))$perf %>%
  filter(rmse < min(rmse) + sd(rmse),
         n_feats < 200) %>%
  arrange(rmse) %>%
  slice_head(n=1)

# generate importance table
icd9_lookup <- tibble(icd9cm_hierarchy) %>% mutate_all(str_to_lower)
icd10_lookup <- tibble(icd10cm2016) %>%  mutate_all(str_to_lower)

perf$features %>%
  mutate(Importance=num(Importance,digits=1)) %>%
  slice_head(n=params$n_feats[1]) %>%
  mutate(code=str_replace(Variable,'^icd_','')) %>%
  left_join(icd9_lookup  %>% select(code,long_desc),
            by='code') %>%
  left_join(icd10_lookup %>% select(code,long_desc),
            by='code') %>%
  mutate(long_desc.x=if_else(is.na(long_desc.x) & is.na(long_desc.y),
                             code,long_desc.x),
         Variable=if_else(str_detect(Variable,'^icd_'),
                          glue('icd: {if_else(!is.na(long_desc.x),',
                               'long_desc.x,long_desc.y)}'),
                          Variable),
         Variable=str_replace(Variable,'count_service_','service: '),
         Variable=str_replace(Variable,'count_',''),
         Variable=str_replace(Variable,'_|\\:',''),
         Variable=str_replace(Variable,'^del$','HC: deliri'),
         Variable=str_replace(Variable,'^cam$','HC: cam'),
         Variable=str_replace(Variable,'^ciwa$',glue('HC: ciwa|alcoho|',
                                                   'withdraw|overdos|',
                                                   'detox|tremens|cows')),
         Variable=str_replace(Variable,'^icd','ICD: '),
         Variable=str_replace(Variable,'^geri$','HC: geriatr'),
         Variable=str_replace(Variable,'^psych$','HC: psychiatr'),
         Variable=str_replace(Variable,'^psychmed$',
                              glue('HC: haloperidol|haldol|olanz|',
                                   'symbyax|precedex|dexmedet|',
                                   'seroquel|quetiapine')),
         Variable=str_replace(Variable,'^hep$',
                              glue('HC: hepatit|hepatol|ascit|jaund|',
                                   'cirrh|varices|meld|portal')),
         Variable=str_replace(Variable,'^numallergies$',
                              'Number of allergies on admission'),
         Variable=str_replace(Variable,'^inf$',
                              glue('HC: antibiot|bacteremi|mssa|mrsa|sepsi')),
         Variable=str_replace(Variable,'^enceph$','HC: enceph'),
         Variable=str_replace(Variable,'^tox$','HC: toxic'),
         Variable=str_replace(Variable,'^confms$',
                              glue('MS: confus|disorient|alter')),
         Variable=str_replace(Variable,'^service orthopaedics$',
                              'Admission service: orthopedics'),
         Variable=str_replace(Variable,'^service cardiothoracic$',
                              'Admission service: cardiothoracic'),
         Variable=str_replace(Variable,'^service surgery$',
                              'Admission service: surgery'),
         Variable=str_replace(Variable,'^service medicine$',
                              'Admission service: medicine'),
         Variable=str_replace(Variable,'year','Admission year'),
         Variable=str_replace(Variable,'monthacademic',
                              'Admission academic month'),
         Variable=str_replace(Variable,'^julyaugust$',
                              'July or august admission'),
         Variable=str_replace(Variable,'^exf$','DDi: extend|servic'),
         Variable=str_replace(Variable,'^enceph$','HC: enceph'),
         Variable=str_replace(Variable,'^psych$','HC: psychiatr'),
         Variable=str_replace(Variable,'^manic$',glue('HC: manic|mania|bipol|',
                                                    'lithium|lumateper|',
                                                    'caplyta|idone|latuda|',
                                                    'depakote|abilify|',
                                                    'saphris|lamictal|',
                                                    'aripipr|lamotrig')), 
         Variable=str_replace(Variable,'^nummeds$',
                              'Number of medications on admission'),
         Variable=str_replace(Variable,'^dischargedate$','Discharge date'),
         Variable=str_replace(Variable,
                              'f1011','ICD: alcohol abuse, in remission'),
         Variable=str_replace(Variable,'^lenpmhx$',
                              'Length of past medical history'),
         Variable=str_replace(Variable,'^confhc$',glue('HC: confus|disorient|',
                                                     'waxing|sundowni|',
                                                     'sun downi|restrain|',
                                                     'halluc')),
         Variable=str_replace(Variable,'^age$','Age on admission'),
         Variable=str_replace(Variable,'^home$','DDi: home'),
         Variable=str_replace(Variable,'^nsurg$','HC: neurosurg|craniot'),
         Variable=str_replace(Variable,'^ao3hc$',
                              'HC: ao/oriented x 3/three'),
         Variable=str_replace(Variable,'^deldd$','DDx: deliri'),
         Variable=str_replace(Variable,'^alz$',
                              glue('HC: alzh|brexpip|donepe|galant|',
                                   'memant|rivastig|aricept|exelon|razadyne')),
         Variable=if_else(str_detect(Variable,'g928'),
                          'ICD: other toxic encephalopathy',
                          Variable),
         Variable=if_else(str_detect(Variable,'z20822'),
                          glue('ICD: asymptomatic patient exposed to ',
                               'actual or suspected COVID'),
                          Variable),
         Variable=str_replace(Variable,'^hosp$','DDi: hospice|expir'),
         Variable=str_replace(Variable,'^sex$','Sex'),
         Variable=str_replace(Variable,'^los$','Length of stay'),
         Variable=str_replace_all(Variable,'\\|',', ')) %>%
  select(`Feature (Count)`=Variable,`Importance (Gini)`=Importance) %>%
  flextable() %>%
  flextable::align(align='center',j=2,part='all') %>%
  flextable::align(align='left',j=1,part='all') %>%
  width(j=1, width = 5) %>%
  width(j=2, width = 2) %>%
  save_as_docx(path=file.path(path,'tbls','imp_tbl.docx'))

p1 <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  left_join(read_csv(file.path(path,'data_out','02a_icd_tbl_pub.csv')),
            by='id') %>%
  filter(set == 'train') %>%
  select(contains('full'),contains('pseudo'),contains('icd')) %>%
  pivot_longer(everything(),values_to = 'label') %>%
  group_by(name,label) %>%
  reframe(n=n()) %>%
  mutate(fr=factor(str_extract(name,'fr[0-9]+') %>% parse_number()),
         th=str_extract(name,'th[0-9]+') %>% parse_number(),
         th=if_else(is.na(th),0,th),
         th=case_when(
           name == 'label_icd' ~ 'ICD',
           name == 'label_pub_icd' ~ 'ICD*',
           name == 'label_fullexpert_fr100' ~ 'Liberal',
           TRUE ~ as.character(th)),
         th=factor(th,levels=c('60','70','80','90','Liberal','ICD','ICD*')),
         label=factor(case_when(
           label == -1 | is.na(label) ~ 'Unlabeled',
           label == 1 ~ 'Positive',
           label == 0 ~ 'Negative'),
           levels=c('Positive','Negative','Unlabeled'),ordered=TRUE),
         set=if_else(str_detect(name,'fullexpert'),'liberal','pseudo')) %>%
  select(-name) %>%
  filter((set == 'pseudo' & fr == 100) | 
           str_detect(th,'ICD') |
           (set == 'liberal' & fr == 100)) %>%
  select(-set) %>%
  ggplot(aes(x=th,y=n,fill=label)) +
  geom_col(position='fill',color='black',alpha=0.5,width=1) +
  # scale_fill_manual(values=c('gray','red','lightblue')) +
  scale_x_discrete(drop = TRUE) +
  theme_classic() +
  theme(legend.position='none') +
  labs(x='Self-Training Threshold',
       y='',
       title='',
       fill='') +
  scale_fill_brewer(type='qual',palette='Set1') 

p2 <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  filter(set == 'train') %>%
  select(contains('full'),contains('pseudo')) %>%
  pivot_longer(everything(),values_to = 'label') %>%
  group_by(name,label) %>%
  reframe(n=n()) %>%
  mutate(fr=factor(str_extract(name,'fr[0-9]+') %>% parse_number()),
         th=str_extract(name,'th[0-9]+') %>% parse_number(),
         th=factor(if_else(is.na(th),0,th)),
         label=factor(case_when(
           label == -1 ~ 'Unlabeled',
           label == 1 ~ 'Positive',
           label == 0 ~ 'Negative'),
           levels=c('Positive','Negative','Unlabeled'),ordered=TRUE),
         set=if_else(str_detect(name,'fullexpert'),'liberal','pseudo')) %>%
  select(-name) %>%
  filter(set == 'pseudo',
         th == 90) %>%
  ggplot(aes(x=fr,y=n,fill=label)) +
  geom_col(position='fill',color='black',alpha=0.5,width=1) +
  # scale_fill_manual(values=c('gray','red','lightblue')) +
  scale_x_discrete(drop = TRUE) +
  theme_classic() +
  theme(legend.position='bottom') +
  labs(x='Proportion of Expert Labels for Self-Training (%)',
       y='',
       title='',
       fill='') +
  scale_fill_brewer(type='qual',palette='Set1') 
fig <- grid.arrange(p1,p2,ncol=1)
ggsave(plot=fig,file.path(path,'figs','pseudo_dist.png'),width=5,height=7)




comps <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  filter(set == 'train') %>%
  left_join(read_csv(file.path(path,'data_out','02a_icd_tbl_pub.csv')),
            by='id') %>% 
  select(contains('icd'),label_fullexpert_fr100,matches('pseudo.*fr100')) %>%
  filter_all(all_vars(!is.na(.) & . != -1))

# Function to calculate Jaccard similarity
jaccard_similarity <- function(x, y) {
  intersection <- sum(x & y)
  union <- sum(x | y)
  return(intersection / union)
}

jaccard_matrix <- matrix(NA, ncol = ncol(comps), nrow = ncol(comps))
colnames(jaccard_matrix) <- colnames(comps)
rownames(jaccard_matrix) <- colnames(comps)

for (i in 1:ncol(comps)) {
  for (j in 1:ncol(comps)) {
    jaccard_matrix[i, j] <- jaccard_similarity(comps[, i], comps[, j])
  }
}

# Convert matrix to long format for ggplot
fig <- reshape2::melt(jaccard_matrix) %>%
  mutate(Var1 = case_when(
    Var1 == 'label_icd' ~ 'Label: ICD',
    Var1 == 'label_pub_icd' ~ 'Label: ICD*',
    Var1 == 'label_fullexpert_fr100' ~ 'Label: Liberal',
    TRUE ~ paste('Pseudo Threshold:',as.character(str_extract(Var1,'th[0-9]+') %>% parse_number()))),
    Var2 = case_when(
      Var2 == 'label_icd' ~ 'Label: ICD',
      Var2 == 'label_pub_icd' ~ 'Label: ICD*',
      Var2 == 'label_fullexpert_fr100' ~ 'Label: Liberal',
      TRUE ~ paste('Pseudo Threshold:',as.character(str_extract(Var2,'th[0-9]+') %>% parse_number()))),
    Var1 = factor(Var1,levels=c('Pseudo Threshold: 60','Pseudo Threshold: 70',
                                'Pseudo Threshold: 80','Pseudo Threshold: 90',
                                'Label: Liberal','Label: ICD','Label: ICD*')),
    Var2 = factor(Var2,levels=c('Pseudo Threshold: 60','Pseudo Threshold: 70',
                                'Pseudo Threshold: 80','Pseudo Threshold: 90',
                                'Label: Liberal','Label: ICD','Label: ICD*'))) %>%
  ggplot(aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_distiller(type='seq',direction = 1,limits=c(0,1)) +
  labs(x = "",
       y = "",
       fill = "") +
  theme_classic() +
  coord_fixed() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
ggsave(plot=fig,file.path(path,'figs','labs_jaccard.png'),width=7,height=5)


comps <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  filter(set == 'train') %>%
  left_join(read_csv(file.path(path,'data_out','02a_icd_tbl_pub.csv')),
            by='id') %>% 
  select(id,contains('icd'),label_fullexpert_fr100,matches('pseudo.*fr100')) %>%
  filter_all(all_vars(!is.na(.) & . != -1)) %>%
  left_join(read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
              select(id,starts_with('count_')),by='id') %>%
  mutate_at(vars(starts_with('count_')),~if_else(. > 0,1,0)) %>%
  select(-id)

jaccard_matrix <- matrix(NA, ncol = ncol(comps), nrow = ncol(comps))
colnames(jaccard_matrix) <- colnames(comps)
rownames(jaccard_matrix) <- colnames(comps)

for (i in 1:ncol(comps)) {
  for (j in 1:ncol(comps)) {
    jaccard_matrix[i, j] <- jaccard_similarity(comps[, i], comps[, j])
  }
}

# Convert matrix to long format for ggplot
df <- reshape2::melt(jaccard_matrix) %>%
  mutate(Var1 = case_when(
    Var1 == 'label_icd' ~ 'Label: ICD',
    Var1 == 'label_pub_icd' ~ 'Label: ICD*',
    Var1 == 'label_fullexpert_fr100' ~ 'Label: Liberal',
    str_detect(Var1,'count')  & !str_detect(Var1,'service')~ str_replace(Var1,'count_',''),
    str_detect(Var1,'service') ~ str_replace(Var1,'count_service','Service: '),
    str_detect(Var1,'pseudo') ~ paste("Pseudo Threshold: ",str_extract(Var1,'th[0-9]+') %>% parse_number()),
    TRUE ~ Var1),
    Var2 = case_when(
      Var2 == 'label_icd' ~ 'Label: ICD',
      Var2 == 'label_pub_icd' ~ 'Label: ICD*',
      Var2 == 'label_fullexpert_fr100' ~ 'Label: Liberal',
      str_detect(Var2,'count') & !str_detect(Var2,'service') ~ str_replace(Var2,'count_',''),
      str_detect(Var2,'service') ~ str_replace(Var2,'count_service','Service: '),
      str_detect(Var2,'pseudo') ~ paste("Pseudo Threshold: ",str_extract(Var2,'th[0-9]+') %>% parse_number()),
      TRUE ~ Var2)) %>%
  mutate(Var1 = str_replace(Var1,'_hc',''),
         Var2 = str_replace(Var2,'_hc',''),
         Var1 = str_replace(Var1,'med$','medication'),
         Var2 = str_replace(Var2,'med$','medication'),
         Var1 = str_replace(Var1,'inf','infection'),
         Var2 = str_replace(Var2,'inf','infection'),
         Var1 = if_else(str_detect(Var1,'prob'),
                        glue("Problem List: {str_replace(Var1,'prob','')}"),
                        Var1),
         Var2 = if_else(str_detect(Var2,'prob'),
                        glue("Problem List: {str_replace(Var2,'prob','')}"),
                        Var2),
         Var1 = if_else(str_detect(Var1,'ms'),
                        glue("Mental Status: {str_replace(Var1,'ms','')}"),
                        Var1),
         Var2 = if_else(str_detect(Var2,'ms'),
                        glue("Mental Status: {str_replace(Var2,'ms','')}"),
                        Var2),
         Var1 = if_else(str_detect(Var1,'dd'),
                        glue("Discharge Diagnosis: {str_replace(Var1,'dd','')}"),
                        Var1),
         Var2 = if_else(str_detect(Var2,'dd'),
                        glue("Discharge Diagnosis: {str_replace(Var2,'dd','')}"),
                        Var2),
         Var1 = str_replace(Var1,'alz','alzheimer'),
         Var2 = str_replace(Var2,'alz','alzheimer'),
         Var1 = str_replace(Var1,'conf','confused'),
         Var2 = str_replace(Var2,'conf','confused'),
         Var1 = str_replace(Var1,'psych','psychiatric'),
         Var2 = str_replace(Var2,'psych','psychiatric'),
         Var1 = str_replace(Var1,'nsurg','neurosurgery'),
         Var2 = str_replace(Var2,'nsurg','neurosurgery'),
         Var1 = str_replace(Var1,'del','delirium'),
         Var2 = str_replace(Var2,'del','delirium'),
         Var1 = str_replace(Var1,'hep$','hepatic'),
         Var2 = str_replace(Var2,'hep$','hepatic'),
         Var1 = str_replace(Var1,'geri$','geriatric'),
         Var2 = str_replace(Var2,'geri$','geriatric'),
         Var1 = str_replace(Var1,'exf','Discharge Disposition: services'),
         Var2 = str_replace(Var2,'exf','Discharge Disposition: services'),
         Var1 = str_replace(Var1,'toxenceph','toxic encephalopathy'),
         Var2 = str_replace(Var2,'toxenceph','toxic encephalopathy'),
         Var1 = str_replace(Var1,'tox$','toxic'),
         Var2 = str_replace(Var2,'tox$','toxic'),
         Var1 = str_replace(Var1,'home','Discharge Disposition: home'),
         Var2 = str_replace(Var2,'home','Discharge Disposition: home'),
         Var1 = str_replace(Var1,'hepenceph',' hepatic encephalopathy'),
         Var2 = str_replace(Var2,'hepenceph',' hepatic encephalopathy'),
         Var1 = if_else(str_detect(Var1,'\\:'),Var1,paste('Hospital Course:',Var1)),
         Var2 = if_else(str_detect(Var2,'\\:'),Var2,paste('Hospital Course:',Var2)),
         Var1 = str_replace(Var1,'enceph_','encephalopathy'),
         Var2 = str_replace(Var2,'enceph_','encephalopathy'),
         Var1 = str_replace(Var1,'_',' '),
         Var2 = str_replace(Var2,'_',' '),
         Var1 = str_squish(Var1),
         Var2 = str_squish(Var2)) %>%
  filter(!str_detect(Var1,'Mental Status|Problem List|Service'),
         !str_detect(Var2,'Mental Status|Problem List|Service'))

fig <- df %>%
  ggplot(aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_distiller(type='seq',direction = 1) +
  labs(x = "",
       y = "",
       fill = "") +
  theme_classic() +
  coord_fixed() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
ggsave(plot=fig,file.path(path,'figs','labs_feats_jaccard.png'),width=7,height=6)








read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  filter(set == 'train',
         label == -1) %>%
  select(contains('full'),contains('pseudo')) %>%
  pivot_longer(everything(),values_to = 'label') %>%
  group_by(name,label) %>%
  reframe(n=n()) %>%
  mutate(fr=factor(str_extract(name,'fr[0-9]+') %>% parse_number()),
         th=str_extract(name,'th[0-9]+') %>% parse_number(),
         th=factor(if_else(is.na(th),0,th)),
         label=factor(if_else(label == -1,'Unlabeled',as.character(label)),
                      levels=c('Unlabeled','1','0'),ordered=TRUE),
         set=if_else(str_detect(name,'fullexpert'),'liberal','pseudo')) %>%
  select(-name)  %>%
  filter(set == 'pseudo',
         fr == '100') %>%
  select(-set) %>%
  pivot_wider(id_cols = th,names_from=label,values_from=n) %>%
  arrange(desc(th)) %>%
  select('Thres.'=th,
         Unlabeled,`0`,`1`) %>%
  flextable() %>%
  autofit() %>%
  flextable::align(align='center',j=1:4,part='all') %>%
  fontsize(size=9,part='body') %>%
  fontsize(i=1,size=11,part='header') %>%
  add_footer_lines(
    glue('Pseudo-label distribution of unlabeled admissions.')) %>%
  set_table_properties(layout = "autofit") %>%
  save_as_docx(path=file.path(path,'tbls','pseudolabs.docx'))

tbl <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  filter(set == 'train',
         label != -1) %>%
  select(contains('full'),contains('pseudo')) %>%
  pivot_longer(everything(),values_to = 'label') %>%
  group_by(name,label) %>%
  reframe(n=n()) %>%
  mutate(fr=factor(str_extract(name,'fr[0-9]+') %>% parse_number()),
         th=str_extract(name,'th[0-9]+') %>% parse_number(),
         th=factor(if_else(is.na(th),0,th)),
         label=factor(if_else(label == -1,'Unlabeled',as.character(label)),
                      levels=c('Unlabeled','1','0'),ordered=TRUE),
         set=if_else(str_detect(name,'fullexpert'),'liberal','pseudo')) %>%
  select(-name)  %>% 
  filter(th %in% c(0,90)) %>%
  select(-th) %>%
  pivot_wider(id_cols = c(set,fr),names_from=label,values_from=n) %>%
  arrange(set,desc(fr)) %>%
  select('Frac.'=fr,set,
         -Unlabeled,
         `0`,`1`) %>%
  pivot_wider(names_from=set,values_from=c(`0`,`1`)) %>%
  select(`Frac.`,`0_liberal`,`1_liberal`,`0_pseudo`,`1_pseudo`) %>%
  flextable() %>%
  set_header_labels(`0_liberal` = "0", `1_liberal` = "1", 
                    `0_pseudo` = "0", `1_pseudo` = "1") %>%
  autofit() %>%
  add_header_row(top=TRUE,values=c('','Liberal','','Pseudo','')) %>%
  merge_at(i=1,j=2:3,part='header') %>%
  merge_at(i=1,j=4:5,part='header') %>%
  flextable::align(align='center',j=1:5,part='all') %>%
  fontsize(size=9,part='body') %>%
  fontsize(i=1,size=11,part='header') %>%
  add_footer_lines(glue('Pseudo-label distribution of expert labeled ',
                        'admissions after fractionating.')) %>%
  set_table_properties(layout = "autofit") %>%
  save_as_docx(path=file.path(path,'tbls','pseudolabs_fr.docx'))



llm_res <- read_csv(file.path(path,'from_python','run_cw_0.csv'))
border_style <- officer::fp_border(color='black', width=1)
llm_res %>%
  rename(fkw=filter_keywords,b_accuracy=b_acc,wd=w_decay) %>%
  mutate(th=as.integer(th),
         nb=as.integer(eff_n_batch),
         across(where(is.double) & ! lr, ~num(.x, digits = 2)),  
         fkw=case_when(
           fkw == 1 ~ 'True',
           fkw == 0 ~ 'False'
         ),
         lr=scientific(lr, digits = 2)) %>%
  arrange(desc(b_accuracy)) %>%
  select('Filt. Kw.'=fkw,
         'Thres.'=th,
         'L.Rate'=lr,
         'Wt.Dec.'=wd,
         'Ef.Batch'=nb,
         'Class Wt.'=cw,
         'Bal. Acc.'=b_accuracy) %>%
  flextable() %>%
  autofit() %>%
  flextable::align(align='center',j=1:6,part='all') %>%
  fontsize(size=9,part='body') %>%
  fontsize(i=1,size=11,part='header') %>%
  set_table_properties(layout = "autofit") %>%
  save_as_docx(path = file.path(path, 'tbls', 'lf_part_ep1.docx'))

llm_res <- read_csv(file.path(path,'from_python','run_cw_1.csv'))
border_style <- officer::fp_border(color='black', width=1)
llm_res %>%
  rename(fkw=filter_keywords,b_accuracy=b_acc,wd=w_decay) %>%
  mutate(th=as.integer(th),
         nb=as.integer(eff_n_batch),
         across(where(is.double) & ! lr, ~num(.x, digits = 2)),  
         fkw=case_when(
           fkw == 1 ~ 'True',
           fkw == 0 ~ 'False'
         ),
         lr=scientific(lr, digits = 2)) %>%
  arrange(desc(b_accuracy)) %>%
  select('Filt. Kw.'=fkw,
         'Thres.'=th,
         'L.Rate'=lr,
         'Wt.Dec.'=wd,
         'Ef.Batch'=nb,
         'Class Wt.'=cw,
         'Bal. Acc.'=b_accuracy) %>%
  flextable() %>%
  autofit() %>%
  flextable::align(align='center',j=1:6,part='all') %>%
  fontsize(size=9,part='body') %>%
  fontsize(i=1,size=11,part='header') %>%
  set_table_properties(layout = "autofit") %>%
  save_as_docx(path = file.path(path, 'tbls', 'lf_full_ep1.docx'))





llm_res <- read_csv(file.path(path,'from_python','run_cw_final_results.csv.gz'))

llm_res %>%
  mutate(lab=case_when(
    lab == 'full' ~ 'Liberal',
    lab == 'only' ~ 'Strict',
    lab == 'pseudo' ~ 'Pseudo'
  )) %>%
  filter(fr > 20,
         set != 'filtered') %>%
  mutate(set=case_when(
    set == 'expert' ~ 'Expert',
    set == 'icd' ~ 'ICD'
  )) %>%
  ggplot(aes(f1,b_accuracy,color=set)) +
  geom_point(alpha=0.9,size=3,shape='O') +
  theme_classic() +
  theme(aspect.ratio = 1,
        legend.position = 'bottom',
        panel.border = element_rect(color='black',fill=NA,linewidth=0.5),
        plot.title = element_text(face = "plain", size = 10),
        panel.grid.major = element_line(color = "grey80", size = 0.5)) +
  facet_grid(lab ~ fr) +
  scale_color_viridis_d(option='H',begin=.7) +
  labs(color='',x='F1',y='Balanced Accuracy',
       title='Percentage of Fractionated Data (%)') 

baseline <- read_csv(file.path(path,'res','15_lasso_results.csv.gz')) %>%
  filter(!is.na(b_acc),
         fraction == 100 | is.na(fraction)) %>%
  mutate(threshold=if_else(is.na(threshold),'',as.character(threshold)),
         w = as.integer(w),
         prop1=num(prop1*100,label='%'),
         lab=case_when(
           label == 'fullexpert' ~ 'Liberal',
           label == 'onlyexpert' ~ 'Strict',
           label == 'icd' ~ 'ICD',
           label == 'pseudo' ~ 'Pseudo'
         )) %>%
  select(-fn,-fraction) %>%
  arrange(label,desc(b_acc),desc(f1)) %>%
  group_by(label) %>%
  slice_head(n=1) %>%
  ungroup() %>%
  filter(label != 'icd') %>%
  select(b_accuracy=b_acc,
         f1,
         lab) %>%
  mutate(set = 'Baseline Model')

llm1 <-
  llm_res %>%
  mutate(lab=case_when(
    lab == 'full' ~ 'Liberal',
    lab == 'only' ~ 'Strict',
    lab == 'pseudo' ~ 'Pseudo',
  ),set=case_when(
    set == 'filtered' ~ 'Filtered',
    set == 'expert' ~ 'Expert Labels',
    set == 'icd' ~ 'ICD Labels'
  ),cw = as.factor(cw)) %>%
  filter(fr == 100,
         th == 90,
         set != 'filtered',
         cw == 5,
         set != 'Filtered',
         !(pl %in% c(1,3) & lab == 'Pseudo'),
         fkw == 1) %>%
  bind_rows(baseline) %>%
  ggplot(aes(f1,b_accuracy,color=lab,shape=set)) +
  geom_point(alpha=0.5,size=5) +
  theme_classic() +
  theme(aspect.ratio = 1,
        legend.position = 'bottom',
        legend.box='Vertical',
        legend.spacing.x=unit(0.0,'cm'),
        plot.margin = unit(c(1, 1, 1, -5),"cm"),
        legend.spacing.y=unit(-0.2,'cm'),
        panel.border = element_rect(color='black',fill=NA,linewidth=0.5),
        plot.title = element_text(face = "plain", size = 10),
        panel.grid.major = element_line(color = "grey80", size = 0.5)) +
  scale_color_brewer(type='qual',palette='Set1') +
  scale_shape_manual(values=c(3,16,17)) + 
  # facet_grid(~set) +
  xlim(0,1) + ylim(0.4,1) +
  labs(color='',shape='',
       x='F1',y='Balanced Accuracy',
       title='')

llm2 <-
  llm_res %>%
  mutate(lab=case_when(
    lab == 'full' ~ 'Liberal',
    lab == 'only' ~ 'Strict',
    lab == 'pseudo' ~ 'Pseudo'
  ),set=case_when(
    set == 'filtered' ~ 'Filtered',
    set == 'expert' ~ 'Expert',
    set == 'icd' ~ 'ICD'
  ),cw = as.factor(cw)) %>%
  filter(th == 90,
         set != 'filtered',
         cw == 5,
         fkw == 1,
         !(pl %in% c(2,3) & lab == 'Pseudo'),
         set == 'Expert',
         lab != 'Strict') %>%
  mutate(fr=as.factor(fr)) %>%
  ggplot(aes(f1,b_accuracy,color=lab)) +
  geom_point(alpha=0.5,size=5) +
  theme_classic() +
  theme(aspect.ratio = 1,
        legend.position = 'none',
        legend.box='Vertical',
        plot.margin = unit(c(1, 1, 1, -5),"cm"),
        legend.spacing.x=unit(0.0,'cm'),
        legend.spacing.y=unit(-0.2,'cm'),
        panel.border = element_rect(color='black',fill=NA,linewidth=0.5),
        plot.title = element_text(face = "plain", size = 10),
        panel.grid.major = element_line(color = "grey80", size = 0.5)) +
  scale_color_brewer(type='qual',palette='Set1') +
  facet_wrap(~fr) + 
  xlim(.7,1) + ylim(.7,1) +
  labs(color='',shape='',
       x='F1',y='Balanced Accuracy',
       title='Fraction of Expert Labels for ST (%)')

fig <- grid.arrange(llm1,llm2,ncol=2,bottom='',
                    widths=c(1,.5))
ggsave(plot = fig,
       file.path(path,'figs','lf_res_final.png'),width=10,height=5,dpi = 300)

fig_th1 <- llm_res %>%
  mutate(lab=case_when(
    lab == 'full' ~ 'Liberal',
    lab == 'only' ~ 'Strict',
    lab == 'pseudo' ~ 'Pseudo'
  ),set=case_when(
    set == 'filtered' ~ 'Filtered',
    set == 'expert' ~ 'Expert',
    set == 'icd' ~ 'ICD'
  ),cw = as.factor(cw)) %>%
  filter(fr == 100,
         set != 'filtered',
         cw == 5,
         fkw == 1,
         !(pl %in% c(2,3) & lab == 'Pseudo'),
         set == 'Expert',
         lab == 'Pseudo') %>%
  mutate(th=as.factor(th)) %>%
  ggplot(aes(f1,b_accuracy,color=th)) +
  geom_point(alpha=0.5,size=5) +
  theme_classic() +
  theme(aspect.ratio = 1,
        legend.position = 'bottom',
        legend.box='Vertical',
        legend.spacing.x=unit(0.0,'cm'),
        plot.margin = unit(c(1, 1, 1, -5),"cm"),
        legend.spacing.y=unit(-0.2,'cm'),
        panel.border = element_rect(color='black',fill=NA,linewidth=0.5),
        plot.title = element_text(face = "plain", size = 10),
        panel.grid.major = element_line(color = "grey80", size = 0.5)) +
  xlim(0.5,1) + ylim(0.5,1) +
  scale_color_brewer(type='qual',palette='Set1') +
  labs(x='F1',y='Balanced Accuracy',
       color='ST Threshold')


fig_th2 <- llm_res %>%
  mutate(lab=case_when(
    lab == 'full' ~ 'Liberal',
    lab == 'only' ~ 'Strict',
    lab == 'pseudo' ~ 'Pseudo'
  ),set=case_when(
    set == 'filtered' ~ 'Filtered',
    set == 'expert' ~ 'Expert',
    set == 'icd' ~ 'ICD'
  ),
  cw = as.factor(cw),
  th = as.factor(th)) %>%
  filter(set == 'Expert',
         fr == 100,
         pl == 1,
         cw == 5,
         fkw == 1,
         lab == 'Pseudo') %>%
  mutate(FPR = fp/(tn + fp),
         FNR = fn/(tp + fn)) %>%
  ggplot(aes(FNR,FPR,color=th)) +
  geom_point(alpha=0.7,size=5) +
  theme_classic() +
  theme(aspect.ratio = 1,
        legend.position = 'none',
        legend.box='Vertical',
        plot.margin = unit(c(1, 1, 1, -5),"cm"),
        legend.spacing.x=unit(0.0,'cm'),
        legend.spacing.y=unit(-0.2,'cm'),
        panel.border = element_rect(color='black',fill=NA,linewidth=0.5),
        plot.title = element_text(face = "plain", size = 10),
        panel.grid.major = element_line(color = "grey80", size = 0.5)) +
  xlim(0,0.5) + ylim(0,0.5) +
  scale_color_brewer(type='qual',palette='Set1') +
  labs(color='ST Threshold')

fig <- grid.arrange(fig_th1,fig_th2,ncol=2,bottom='',
                    widths=c(1,.5))
ggsave(plot = fig,
       file.path(path,'figs','th_fpr_fnr_th.png.png'),
       width=10,height=5,dpi = 300)


border_style <- officer::fp_border(color='black', width=1)
llm_res %>%
  filter(set != 'filtered') %>%
  mutate(th=as.integer(th),
         fr=as.integer(fr),
         nb=as.integer(nb),
         cw=as.integer(cw),
         across(where(is.double) & ! lr, ~num(.x, digits = 2)),  
         prop_pred_positive =num(prop_pred_positive *100,label='%'),
         prop_true_positive =num(prop_true_positive *100,label='%'),
         fkw=case_when(
           fkw == 1 ~ 'True',
           fkw == 0 ~ 'False'
         ),
         pl=case_when(
           pl == 1 ~ 'FT',
           pl == 2 ~ 'FT+PT',
           pl == 3 ~ 'FT+PT+T'
         ),
         set=case_when(
           set == 'icd' ~ 'ICD',
           set == 'expert' ~ 'Exp'
         ),
         lab=case_when(
           lab == 'full' ~ 'Liberal',
           lab == 'only' ~ 'Strict',
           lab == 'pseudo' ~ 'Pseudo'
         ),
         lr=scientific(lr, digits = 2)) %>%
  select(-ls) %>%
  mutate(lab=factor(lab,levels=c('Pseudo','Liberal','Strict'),ordered=TRUE),
         set=factor(set,levels=c('Exp','ICD'),ordered=TRUE),
         th=if_else(lab == 'Pseudo',as.character(th),'')) %>%
  arrange(lab,set,desc(fkw),pl,desc(b_accuracy)) %>%
  select('Lab.'=lab,
         'Set'=set,
         'Filt. Kw.'=fkw,
         'Thres.'=th,
         'Frac.'=fr,
         'PL'=pl,
         'L.Rate'=lr,
         'Wt.Dec.'=wd,
         'Ef.Batch'=nb,
         'Class Wt.'=cw,
         'Acc.'=accuracy,
         'Bal. Acc.'=b_accuracy,
         'AUC'=auc,
         'Prec.'=precision,
         'Rec.'=recall ,
         'Spec.'=spec,
         'F1'=f1,
         'Pred.1(%)'=prop_pred_positive,
         'True.1(%)'=prop_true_positive) %>%
  filter(`Thres.` %in% c('90',''),
         `Class Wt.` %in% c(5,50,100)) %>%
  flextable() %>%
  autofit() %>%
  flextable::align(align='center',j=1:19,part='all') %>%
  fontsize(size=7,part='body') %>%
  fontsize(i=1,size=8,part='header') %>%
  set_table_properties(layout = "autofit") %>%
  save_as_docx(path = file.path(path, 'tbls', 'lf.docx'),
               pr_section = prop_section(
                 page_size = page_size(orient = "landscape"),
                 type = "continuous"
               ))

border_style <- officer::fp_border(color='black', width=1)
llm_res %>%
  filter(lab == 'pseudo',
         set == 'expert',
         fkw == 1,
         th == 90,
         cw == 5,
         fr == 100,
         pl == 1) %>%
  mutate(model = 'Longformer') %>%
  select(model,accuracy,b_accuracy,f1,auc,precision,recall) %>%
  bind_rows(read_csv(file.path(path,'from_python','results_llama_cw5.csv')) %>%
              filter(`...1` == 'expert') %>%
              mutate(model = 'Llama') %>%
              select(model,accuracy,b_accuracy,f1,auc,precision,recall)) %>%
  arrange(desc(b_accuracy)) %>%
  select('Model'=model,
         'Acc.'=accuracy,
         'Bal. Acc.'=b_accuracy,
         'AUC'=auc,
         'Prec.'=precision,
         'Rec.'=recall ,
         'F1'=f1) %>%
  mutate(across(where(is.numeric), ~ round(., 4))) %>%
  flextable() %>%
  autofit() %>%
  flextable::align(align='center',j=1:7,part='all') %>%
  fontsize(size=8,part='body') %>%
  fontsize(i=1,size=10,part='header') %>%
  set_table_properties(layout = "autofit") %>%
  save_as_docx(path = file.path(path, 'tbls', 'llm.docx'))


read_csv(file.path(path,'data_out','stm_top_terms.csv.gz')) %>%
  select(Model=mod,
         `P.L.`=pl,
         `Thres.`=th,
         `Feat.`=feature,
         Rank=rank,
         K,
         `Stat.`=stat,
         starts_with('V')) %>%
  rename_with(~str_replace(.,'V',''),starts_with('V')) %>%
  mutate(`Feat.`=case_when(
    `Feat.` == 'fn' ~ 'FN',
    `Feat.` == 'fp' ~ 'FP',
    `Feat.` == 'tn' ~ 'TN',
    `Feat.` == 'tp' ~ 'TP',
    `Feat.` == 'pred_pos' ~ 'PP',
    `Feat.` == 'pred_neg' ~ 'PN',
  ),`P.L.`=case_when(
    `P.L.` == 1 ~ 'FT',
    `P.L.` == 2 ~ 'FT+PT'
  ),`Stat.`=case_when(
    `Stat.` == 'frex' ~ 'FREX',
    `Stat.` == 'freq' ~ 'Freq',
  ),Model=case_when(
    Model == 'lf' ~ 'LF',
    Model == 'lam' ~ 'Llama'
  )) %>% 
  write_csv(file.path(path,'tbls','top_terms.csv'))


library(png)
library(magick)
img1 <- image_read_pdf(file.path(path,'figs','topic1.pdf'))
img1 <- image_crop(img1, "1500x1100+1330+400")
image_write(img1, file.path(path,'figs','topic1_cropped.png'))

img2 <- image_read_pdf(file.path(path,'figs','topic2.pdf'))
img2 <- image_crop(img2, "1500x1100+1330+400")
image_write(img2, file.path(path,'figs','topic2_cropped.png'))

label1 <- textGrob("'Volume-Overload Topic'", 
                   gp=gpar(fontsize=14, fontface="bold"))
label2 <- textGrob("'Renal-Failure Topic'", 
                   gp=gpar(fontsize=14, fontface="bold"))

img1 <- rasterGrob(readPNG(file.path(path,'figs','topic1_cropped.png')), 
                   interpolate = TRUE)
img2 <- rasterGrob(readPNG(file.path(path,'figs','topic2_cropped.png')), 
                   interpolate = TRUE)
fig <- grid.arrange(label1, label2, img1, img2, ncol=2, nrow=2, 
             heights=c(0.1,1),widths=c(1,1),
             layout_matrix=rbind(c(1, 2), c(3, 4)))
ggsave(plot=fig,file.path(path,'figs','top_topic_terms.png'),width=7,height=3.5)


trends1 <- read_csv(file.path(path,'from_python','fkw1_th70_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_unlabeled.csv.gz')) %>%
  mutate(run = 'Threshold 70')
trends2 <- read_csv(file.path(path,'from_python','fkw1_th80_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_unlabeled.csv.gz')) %>%
  mutate(run = 'Threshold 80')
trends3 <- read_csv(file.path(path,'from_python','fkw1_th90_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_unlabeled.csv.gz')) %>%
  mutate(run = 'Threshold 90')
tbl1 <- bind_rows(trends1,trends2,trends3) %>%
  group_by(run,id) %>%
  filter(i == max(i)) %>%
  group_by(run) %>%
  reframe(Positive = sum(preds == 1),
          Negative = sum(preds == 0)) %>%
  mutate(Threshold = str_extract(run,'[0-9]+') %>% parse_number,
         Pipeline = 'FT') %>%
  select(-run)


trends1 <- read_csv(file.path(path,'from_python','fkw1_th90_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_unlabeled.csv.gz')) %>%
  mutate(run = 'FT')
trends2 <- read_csv(file.path(path,'from_python','fkw1_th90_fr100_pl2_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_unlabeled.csv.gz')) %>%
  mutate(run = 'FT + PT')
tbl2 <- bind_rows(trends1,trends2)  %>%
  group_by(run,id) %>%
  filter(i == max(i)) %>%
  group_by(run) %>%
  reframe(Positive = sum(preds == 1),
          Negative = sum(preds == 0)) %>%
  mutate(Threshold = 90) %>%
  rename(Pipeline=run)

bind_rows(tbl1,tbl2) %>%
  distinct() %>%
  select(Pipeline,Threshold, Positive, Negative) %>%
  flextable() %>%
  set_header_labels(Lambda = "L1 λ") %>%
  autofit() %>%
  add_header_row(top=TRUE,
                 values=c('','','Classification','')) %>%
  merge_at(i=1,j=1:2,part='header') %>%
  merge_at(i=1,j=3:4,part='header') %>%
  vline(part='all',j=2,border=border_style) %>%
  flextable::align(align='center',j=1:4,part='all') %>%
  fontsize(size=9,part='body') %>%
  fontsize(i=1:2,size=11,part='header') %>%
  save_as_docx(path=file.path(path,'tbls','unlabeled.docx'))


print_note <- function(s, w = 100) {
  n <- ceiling(str_length(s) / w)
  
  chunks <- str_sub(s, 
                    seq(1, by = w, length.out = n), 
                    seq(w, by = w, length.out = n))
  
  for (chunk in chunks) {
    cat(chunk, "\n")
  }
}

tbl <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  filter(label == -1) %>%
  select(id,hpi_hc,contains('pseudo')) %>%
  select(id,hpi_hc,contains('fr100')) %>%
  mutate(sum = rowSums(select(., contains('pseudo')))) %>%
  filter(sum == 4)

print_note(tbl$hpi_hc[1],100)
print_note(tbl$hpi_hc[5],100)
print_note(tbl$hpi_hc[50],100)
print_note(tbl$hpi_hc[95],100)
print_note(tbl$hpi_hc[120],100)


e1 <- "urine culture revealed infec tion with staphylococcus organism, patient was continue on cipro iv for days total. starting on the patient was noted to have alteration in mental status. geriatric consult was called, their recommend ations were followed. on patient became more agitated, she tried to pull ngt and iv, became severely delirious. patient received dose of haldol with minimal effect, then physical restraints were utili zed, when patient's condition improved, sitter was used for observation."
e2 <- "was delirious postop on narcotics. they were discontinued an d he was placed on ultram and tylenol for pain. his mental status cleared."
e3 <- "continue to monito r nutrition and encourage supplementation with ensures. encephalopathy, resolved while in the icu, t he patient was noted to be delirious and confused, likely in the setting of critical illness and wee k intubation. no sedating meds were on board"
e4 <- "on pod overnight patient became delirious and agitated, requiring haldol."
e5 <- "esolved mild confusion although mo stly oriented, worse . contributing factors likely include pain, opioids, and other hospitalization related stimuli. improved with holding oxycodone"

tibble(Excerpt = c(e1,e2,e3,e4,e5)) %>%
  mutate(i=row_number()) %>%
  select(i,Excerpt) %>%
  flextable() %>%
  flextable::align(align='left',j=1:2,part='all') %>%
  fontsize(size=9,part='body') %>%
  fontsize(i=1,size=11,part='header') %>%
  width(j=1, width = .2) %>%
  width(j=2, width = 5) %>%
  set_header_labels(`i` = "", `Excerpt` = "Excerpt") %>%
  save_as_docx(path=file.path(path,'tbls','pos_exerpts.docx'))

feats %>% 
  select(-id,-set,-label) %>%
  summarise_all(mean) %>%
  pivot_longer(everything()) %>%
  arrange(desc(value)) %>%
  filter(value > 0) %>%
  print(n=Inf)

feats <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
  filter(id %in% tbl$id)

tbl <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  filter(label == -1) %>%
  select(id,hpi_hc,contains('pseudo')) %>%
  select(id,hpi_hc,contains('fr100')) %>%
  mutate(sum = rowSums(select(., contains('pseudo')))) %>%
  filter(sum == 0)

feats <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
  filter(id %in% tbl$id)

feats %>% 
  select(-id,-set,-label) %>%
  summarise_all(mean) %>%
  pivot_longer(everything()) %>%
  arrange(desc(value)) %>%
  filter(value > 0) %>%
  print(n=Inf)


tbl <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  filter(label == -1) %>%
  select(id,hpi_hc,contains('pseudo')) %>%
  select(id,hpi_hc,contains('fr100')) %>%
  mutate(sum = rowSums(select(., contains('pseudo')))) %>%
  filter(sum == -4)

feats <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
  filter(id %in% tbl$id)

feats %>% 
  select(-id,-set,-label) %>%
  summarise_all(mean) %>%
  pivot_longer(everything()) %>%
  arrange(desc(value)) %>%
  filter(value > 0) %>%
  print(n=Inf)



tbl <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  filter(label == -1) %>%
  select(id,hpi_hc,contains('pseudo')) %>%
  select(id,hpi_hc,contains('fr100')) %>%
  select(id,hpi_hc,contains('th90') | contains('th80')) %>%
  filter(label_pseudo_th90_fr100 != label_pseudo_th80_fr100)
  
tbl2 <- tbl %>%
  filter(label_pseudo_th90_fr100 != -1,
         label_pseudo_th80_fr100 != -1) 

print_note(tbl2$hpi_hc[1])
print_note(tbl2$hpi_hc[10])
print_note(tbl2$hpi_hc[35])

feats <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
  filter(id %in% tbl2$id)

feats %>% 
  select(-id,-set,-label) %>%
  summarise_all(mean) %>%
  pivot_longer(everything()) %>%
  arrange(desc(value)) %>%
  filter(value > 0) %>%
  print(n=Inf)
