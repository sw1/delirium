library(tidymodels)
library(glmnet)
library(tidyverse)
library(doParallel)
library(vip)
library(rpart.plot)

get_rules <- function(x){
  y <- matrix('',nrow=nrow(x),ncol=3)
  for (i in 1:nrow(x)){
    d <- as.numeric(x[i,1])
    y[i,3] <- d
    if (d < 0.5){
      y[i,1] <- paste0('v',100-i)
    }else{
      y[i,1] <- paste0('v',nrow(x)-i+1)
    }
    y[i,2] <- gsub('\\s+',' ',
                   paste0(
                     gsub('is','==',x[i,3:ncol(x)]),collapse=' '))
    y[i,2] <- paste0('if_else(',y[i,2],',1,0)')
    
  }
  return(y)
}

new_var <- function(df,v1,v2){
  df %>% mutate(!!rlang::parse_expr(v1) := !!rlang::parse_expr(v2))
}

update_df <- function(df,r){
  for (i in 1:nrow(r)){
    v1 <- r[i,1]
    v2 <- r[i,2]
    df <- new_var(df,v1,v2)
  }
  return(df)
}

get_icds <- function(x){
  z <- NULL
  for (i in 1:nrow(x)){
    r <- unlist(rules[i,])
    y <- r[startsWith(r,'icd_')]
    y <- gsub('icd_','',y)
    z <- c(z,y)
  }
  return(unique(z))
}

process_features <- function(x,n=50){
  feat_n <- nchar(x)
  y <- gsub(' |[[:punct:]]','',x)
  y <- paste0('icd_',y)
  y <- substr(y,1,n)
  y <- paste0(y,'_',feat_n)
  return(y)
}

path <- 'D:\\Dropbox\\embeddings\\delirium'
ss <- 'icd'

ds <- 'count_del'

features <- NULL
m <- 'f_meas'


tree_fit <- read_rds(file.path(path,'data_in',paste0('fit_tree_',ds,sprintf('_%s.rds',ss))))

best_tree <- tree_fit$fit %>% select_best(m)
wf <- tree_fit$wf %>% 
  finalize_workflow(best_tree) %>%
  last_fit(tree_fit$split) %>%
  extract_workflow()

# features <- colnames(wf$fit$fit$fit$model)[-1]
# features_icds_tree <- c(gsub('icd_','',features[!str_detect(features,'^count_|^los_')]),'none')
# features <- c(paste0('icd_',features_icds),features[str_detect(features,'^count_|^los_')])
# rm(tree_fit)
# rm(wf)
# rm(best_tree)


features <- colnames(wf$fit$fit$fit$model)[-1]
features_icds_tree <- gsub('icd_','',features[!str_detect(features,'^count_|^los_')])
rm(tree_fit)
rm(wf)
rm(best_tree)


haobo <- read_rds(file.path(path,'data_in',
                            sprintf('full_icd_tbl_%s.rds',
                                    gsub('_','',ss)))) %>%
  select(-icd_codes) %>%
  rename(icd_codes=icd_sub_chapter) 
# filter(!is.na(label))  # turn off for full data

features_icds <- unique(unlist(haobo$icd_codes))
features_icds <- features_icds[features_icds %in% features_icds_tree]
# features_icds <- features_icds[nchar(features_icds) > 0]
icd_mat <- matrix(0,nrow(haobo),length(features_icds))
colnames(icd_mat) <- paste0('icd_',features_icds)
for (i in 1:nrow(haobo)){
  icds <- na.omit(unlist(haobo$icd_codes[i]))
  icds <- icds[icds %in% features_icds_tree]
  if (length(icds) == 0) next
  icds <- paste0('icd_',icds)
  for (j in seq_along(icds)){
    icd_mat[i,icds[j]] <- icd_mat[i,icds[j]] + 1
  }
}

haobo <- haobo %>% select(-icd_codes)

features_service <- unique(unlist(haobo$service))
service_mat <- matrix(0,nrow(haobo),length(features_service))
colnames(service_mat) <- features_service
for (i in 1:nrow(haobo)){
  servs <- unique(unlist(haobo$service[i]))
  service_mat[i,servs] <- 1
}
colnames(service_mat) <- paste0('count_service_',colnames(service_mat))
service_mat <- service_mat[,colnames(service_mat) != 'count_service_other']
colnames(service_mat)[
  colnames(service_mat) == 'count_service_obstetrics/gynecology'
] <- 'count_service_ob'

haobo <- haobo %>% select(-service)

haobo <- haobo %>%                             # added 1/2/2024
  bind_cols(icd_mat) %>%
  bind_cols(service_mat)

# create metadata
haobo <- haobo %>%
  mutate(los=replace(los,is.infinite(los),NA),
         los=replace(los,is.na(los),mean(los,na.rm=TRUE))) %>%
  mutate(
    count_del=1*str_count(hc,'deliri| cam |cows'),
    count_postdel=1*str_count(hc,'postop deli|postoperative deli'),
    count_del_dd=1*str_count(discharge_diagnosis,'deliri'),
    count_del_ms=1*str_count(mental_status,'deliri'),
    count_del_prob=1*str_count(note,'(\\#\\s?deliri|\\d[:punct:]?\\s+delir)|(\\#\\s?postoperative deliri|\\d[:punct:]?\\s+postoperative delir)|(\\#\\s?postop deliri|\\d[:punct:]?\\s+postop delir)'),
    count_enceph_prob=1*str_count(note,'(\\#\\s?enceph|\\d[:punct:]?\\s+enceph)'),
    count_metenceph_prob=1*str_count(note,'(\\#\\s?met enceph|\\d[:punct:]?\\s+met enceph)|(\\#\\s?metabolic enceph|\\d[:punct:]?\\s+metabolic enceph)'),
    count_hepenceph_prob=1*str_count(note,'(\\#\\s?hep enceph|\\d[:punct:]?\\s+hep enceph)|(\\#\\s?hepatic enceph|\\d[:punct:]?\\s+hepatic enceph)'),
    count_toxenceph_prob=1*str_count(note,'(\\#\\s?tox enceph|\\d[:punct:]?\\s+tox enceph)|(\\#\\s?toxic enceph|\\d[:punct:]?\\s+toxic enceph)'),
    count_conf_hc=1*str_count(hc,'confus|disorient|waxing|sundowni|sun downi|restrain|halluc'),
    count_conf_ms=1*str_count(mental_status,'confus|disorient|alter'),
    count_ao0_ms=1*str_count(mental_status,'((ao|oriented)\\s?x?\\s?(0|zero))|((ao|oriented)\\s?x?\\s?(1|one))|((ao|oriented)\\s?x?\\s?(2|two))'),
    count_ao3_ms=1*str_count(mental_status,'(ao|oriented)\\s?x?\\s?(3|three)'),
    count_ao0_hc=1*str_count(hc,'((ao|oriented)\\s?x?\\s?(0|zero))|((ao|oriented)\\s?x?\\s?(1|one))|((ao|oriented)\\s?x?\\s?(2|two))'),
    count_ao3_hc=1*str_count(hc,'(ao|oriented)\\s?x?\\s?(3|three)'),
    count_exf=1*str_count(discharge_disposition,'extend|servic'),
    count_hosp=1*str_count(discharge_disposition,'hospice|expir'),
    count_home=1*str_count(discharge_disposition,'home'),
    count_homeless=1*str_count(hc,'homeless|shelter'),
    count_schiz=1*str_count(hc,'schizo|delusion|disorganiz|cataton'),
    count_park=1*str_count(hc,'parkins|dopa|duopa|rytary|sinemet'),
    count_alz=1*str_count(hc,'alzh|brexpip|donepe|galant|memant|rivastig|aricept|exelon|razadyne'),
    count_manic=1*str_count(hc,'manic|mania|bipol|lithium|lumateper|caplyta|idone|latuda|depakote|abilify|saphris|lamictal|aripipr|lamotrig'),
    count_enceph=1*str_count(hc,'enceph'),
    count_nsurg=1*str_count(hc,'neurosurg|craniot'),
    count_psych=1*str_count(hc,'psychiatr'),
    count_geri=1*str_count(hc,'geriatr'),
    count_pal=1*str_count(hc,'paliat'),
    count_inf=1*str_count(hc,'antibiot|bacteremi|mssa|mrsa|sepsi'),
    count_psych_med=1*str_count(hc,'haloperidol|haldol|olanz|symbyax|precedex|dexmedet|seroquel|quetiapine'),
    count_ciwa=1*str_count(hc,'ciwa|alcoho|withdraw|overdos|detox|tremens'),
    count_hep=1*str_count(hc,'hepatit|hepatol|ascit|jaund|cirrh|varices|meld|portal'),
    count_tox=1*str_count(hc,'toxic'),
    count_los=los                          # added 1/2/23
  ) 

# create metadata from nursing data
haobo <- haobo %>%
  left_join(read_csv(file.path(path,'data_in','NursingSummariesWithMRN_ID.csv')) %>% 
              select(date_nurs=note_dt,mrn,note_nurs=note_txt) %>%
              mutate(mrn=as.numeric(mrn)),
            by='mrn',relationship='many-to-many') %>%
  mutate(date_nurs=ymd(date_nurs)) %>%
  rowwise() %>%
  mutate(note_overlap=if_else(date_nurs >= date_adm && date_nurs <= date_dc,1,0),
         note_nurs=if_else(note_overlap == 1,note_nurs,NA)) %>%
  ungroup() %>%
  group_by(id) %>%
  mutate(note_nurs=paste(note_nurs,collapse=' '),
         n_nurs=sum(note_overlap,na.rm=TRUE)) %>%
  ungroup() %>%
  select(-note_overlap,-date_nurs) %>%
  distinct() %>%
  mutate(
    count_nurse_del=1*str_count(note_nurs,'deliri| cam |cows'),
    count_nurse_conf_ms=1*str_count(note_nurs,'confus|disorient|waxing|sundowni|sun downi|restrain|halluc'),
    count_nurse_ao0_ms=1*str_count(note_nurs,'((ao|oriented)\\s?x?\\s?(0|zero))|((ao|oriented)\\s?x?\\s?(1|one))|((ao|oriented)\\s?x?\\s?(2|two))'),
    count_nurse_ao3_ms=1*str_count(note_nurs,'(ao|oriented)\\s?x?\\s?(3|three)'),
    count_nurse_pysch_med=1*str_count(note_nurs,'haloperidol|haldol|olanz|symbyax|precedex|dexmedet|seroquel|quetiapine'),
    count_nurse_bp_med=1*str_count(note_nurs,'lithium|lumateper|caplyta|idone|latuda|depakote|abilify|saphris|lamictal|aripipr|lamotrig'),
    count_nurse_alz_med=1*str_count(note_nurs,'brexpip|donepe|galant|memant|rivastig|aricept|exelon|razadyne'),
    count_nurse_wd=1*str_count(note_nurs,'ciwa|alcoho|withdraw|overdos|detox|tremens'),
    count_nurse_jaund=1*str_count(note_nurs,'ascit|jaund|cirrh|varices|meld')
  ) 


haobo <- haobo %>%
  select(!starts_with(c('icd_','count_','los_')),contains(features))

if (ds == 'count_del'){
  haobo_post <- haobo 
}
if (ds == 'count_nodel'){
  haobo_post <- haobo %>%
    select(-count_del) 
}
if (ds == 'binary_del'){
  haobo_post <- haobo %>%
    mutate(across(starts_with('count_'), ~if_else(.x > 0,1,0)))
}
if (ds == 'binary_nodel'){
  haobo_post <- haobo %>%
    select(-count_del) %>%
    mutate(across(starts_with('count_'), ~if_else(.x > 0,1,0)))
}

write_rds(haobo_post,file.path(path,'data_tmp',sprintf('alldat_preprocessed_for_pred_%s.rds',ss)))

haobo_post <- read_rds(file.path(path,'data_tmp',sprintf('alldat_preprocessed_for_pred_%s.rds',ss)))
tree_fit <- read_rds(file.path(path,'data_in',paste0('fit_tree_',ds,sprintf('_%s.rds',ss))))

m <- 'f_meas'
# best_tree <- tree_fit$fit %>% select_by_pct_loss(metric=m,limit=5,tree_depth,desc(min_n))
best_tree <- tree_fit$fit %>% select_by_one_std_err(metric=m,tree_depth,desc(min_n))
wf <- tree_fit$wf %>% 
  finalize_workflow(best_tree) %>%
  last_fit(tree_fit$split) %>%
  extract_workflow()
features <- colnames(wf$fit$fit$fit$model)[-1]

haobo_pred <- haobo_post %>%
  select(id,label,contains(features)) %>%
  mutate(across(everything(), ~replace_na(.x, 0))) %>%
  mutate(label=as.factor(label))

ids <- haobo_pred %>% select(id)
haobo_pred <- haobo_pred %>% select(-id)

preds <- wf %>%
  predict(haobo_pred)

labels <- ids %>% bind_cols(preds) %>% rename(label_tree=.pred_class)
print(table(labels$label_tree))

write_csv(labels,file.path(path,'data_in',paste0('labels_sptree_',ds,sprintf('_%s.csv.gz',ss))),
          col_names=TRUE)


