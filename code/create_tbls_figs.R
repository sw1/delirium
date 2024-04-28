pacman::p_load(tidyverse,glue,gtsummary,flextable,icd.data,tidymodels,
               rpart,rpart.plot)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))


# generate data summary table
tbl1 <- read_rds(file.path(path,'data_in','03_tbl_final.rds')) %>% 
  mutate(sex=if_else(sex==0,'female','male'),
         label=case_when(
           is.na(label) ~ 'Unlabeled',
           label == 1 ~ 'Label: 1',
           label == 0 ~ 'Label: 0',
           TRUE ~ NA)) %>%
  select(label,service,sex,age,los,num_meds,num_allergies,len_pmhx) %>%
  tbl_summary(by=label,
              statistic=list(all_continuous() ~ '{mean} ({sd})',       
                             all_categorical() ~ '{n} ({p}%)'),   
              digits=all_continuous() ~ 1,                             
              type=all_categorical() ~ 'categorical',                
              label=list(                                           
                label ~ 'Label', 
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
  gt::gtsave(file.path(path,'res','demo_tbl.docx'))


# generate rf cv performance tbl
perf <- read_rds(file.path(path,'data_in','08_rf_fs.rds'))

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
         'Acc.'=bacc,
         'Prec.'=prec,
         'Rec.'=rec,
         'F1'=f1,
         'Pred. 1 (%)'=prop1) %>%
  flextable() %>%
  autofit() %>%
  add_header_row(top=TRUE,
                 values=c('','','','RF','Testing','','','','')) %>%
  merge_at(i=1,j=1:3,part='header') %>%
  merge_at(i=1,j=5:9,part='header') %>%
  # merge_at(i=1,j=8:9,part='header') %>%
  vline(part='all',j=3,border=border_style) %>%
  vline(part='all',j=4,border=border_style) %>%
  bg(.,i=~`RMSE` < (min(`RMSE`) + sd(`RMSE`)),
     part='body',bg='lightgray') %>%
  flextable::align(align='center',j=1:9,part='all') %>%
  bold(i=~`Acc.` == max(`Acc.`),j=5,bold=TRUE,part='body') %>%
  bold(i=~`Prec.` == max(`Prec.`),j=6,bold=TRUE,part='body') %>%
  bold(i=~`Rec.` == max(`Rec.`),j=7,bold=TRUE,part='body') %>%
  bold(i=~F1 == max(F1),j=8,bold=TRUE,part='body') %>%
  bold(i=~`RMSE` == min(`RMSE`),j=4,bold=TRUE,part='body') %>%
  bold(i=~`Feat. (N)` < 200,j=1,bold=TRUE,part='body') %>%
  bg(.,i=~`Feat. (N)` < 200 & `RMSE` < 0.2,
     part='body',bg='gray') %>%
  fontsize(size=9, part='body') %>%
  fontsize(i=1:2,size=11,part='header') %>%
  save_as_docx(path=file.path(path,'res','rf_cv.docx'))

params <- read_rds(file.path(path,'data_in','08_rf_fs.rds'))$perf %>%
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
         Variable=str_replace(Variable,'^del$','HC: deliri|cam|cows'),
         Variable=str_replace(Variable,'^ciwa$',glue('HC: ciwa|alcoho|',
                                                   'withdraw|overdos|',
                                                   'detox|tremens')),
         Variable=str_replace(Variable,'^icd','ICD: '),
         Variable=str_replace(Variable,'^geri$','HC: geriatr: '),
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
  save_as_docx(path=file.path(path,'res','imp_tbl.docx'))

features <- perf$features %>%
  mutate(Importance=num(Importance,digits=1)) %>%
  slice_head(n=params$n_feats[1]) %>%
  mutate(code=str_replace(Variable,'^icd_','')) %>%
  left_join(icd9_lookup  %>% select(code,long_desc),
            by='code') %>%
  left_join(icd10_lookup %>% select(code,long_desc),
            by='code') %>%
  mutate(long_desc.x=if_else(is.na(long_desc.x) & is.na(long_desc.y),
                             code,long_desc.x),
         long_desc.x=if_else(is.na(long_desc.x) & !is.na(long_desc.y),
                             long_desc.y,long_desc.x)) %>%
  select(Variable,Importance,Name=long_desc.x)

features_subset <- features %>% 
  pull(Variable) %>% 
  paste('^',.,'$',sep='')

train <- read_rds(
  file.path(path,'data_in','09_alldat_preprocessed_for_pred.rds')) %>%
  filter(set == 'test_expert') %>%
  select(id,label,matches(features_subset)) %>%
  upsamp() %>%
  select(-id) 

set.seed(7)
folds <- vfold_cv(train,v=5,strata=label)
tree_spec <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = get_node_size(params$min_node_perc[1],train)
) %>%
  set_engine('rpart',model=TRUE) %>%
  set_mode('classification')

tuner <- grid_regular(cost_complexity(),
                      tree_depth(c(3,7)),
                      levels=4)

mets <- metric_set(accuracy, sens, yardstick::spec, f_meas, roc_auc)

wf <- workflow() %>%
  add_model(tree_spec) %>%
  add_formula(label ~ .)

fit <- wf %>%
  tune_grid(resamples=folds,
            grid=tuner,
            control=control_grid(verbose=TRUE),
            metrics=mets)

m <- 'f_meas'
best_tree <- fit %>% 
  select_by_one_std_err(metric=m,limit=5,desc(n))

set.seed(7)
tree_spec <- decision_tree(
  cost_complexity = best_tree$cost_complexity[1],
  tree_depth = best_tree$tree_depth[1],
  min_n = get_node_size(params$min_node_perc[1],train)
) %>%
  set_engine('rpart',model=TRUE) %>%
  set_mode('classification')

fit <- workflow() %>%
  add_model(tree_spec) %>%
  add_formula(label ~ .) %>%
  fit(train)

png(file.path(path,'res','tree.png'),width=20,height=18)
fit %>% 
  extract_fit_parsnip() %>%
  .$fit %>%
  rpart.plot(extra=1,cex=0.5,type=3,clip.right.labs=FALSE) 
dev.off()
