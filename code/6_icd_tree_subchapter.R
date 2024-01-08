library(tidymodels)
library(glmnet)
library(tidyverse)
library(doParallel)
library(vip)
library(rpart.plot)
library(icd.data)

all_cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox'
}
source(file.path(path,'code','fxns.R'))

n_folds <- 10
n_sweep <- 5
col_filter <- 10

mod <- 'tree'
subsets <- c('sub_chapter','major')

for(ss in subsets){

  haobo <- read_rds(file.path(path,'data_in',
                              sprintf('full_icd_tbl_%s.rds',
                                      gsub('_','',ss)))) %>%
    select(-icd_codes) %>%
    rename(icd_codes=icd_sub_chapter) %>%
    filter(!is.na(label))  # turn off for full data

  features_icds <- unique(process_features(unlist(haobo$icd_codes)))
  icd_mat <- matrix(0,nrow(haobo),length(features_icds))
  colnames(icd_mat) <- features_icds
  for (i in 1:nrow(haobo)){
    icds <- na.omit(table(unlist(haobo$icd_codes[i])))
    if (length(icds) == 0) next
    names(icds) <- process_features(names(icds))
    for (j in seq_along(icds)){
      icd_mat[i,names(icds)[j]] <- icd_mat[i,names(icds)[j]] + icds[j]
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
  
  haobo <- haobo %>% 
    select(-service) %>%                
    bind_cols(icd_mat) %>%
    bind_cols(service_mat)
  
  # create metadata
  haobo <- create_counts(haobo)
  
  # test split
  n_split <- 50
  
  # icd test for tree
  set.seed(5)
  haobo_test <- haobo %>%
    group_by(label) %>%
    sample_n(n_split) %>%
    ungroup()
  
  haobo <- haobo %>%
    anti_join(haobo_test,by='id')
  
  # upsample smaller class
  set.seed(12)
  n_upsamp <- round(sum(haobo$label == 1)/sum(haobo$label == 0))
  haobo_train <- tibble()
  for (i in 1:n_upsamp){
    haobo_train <- haobo_train %>%
      bind_rows(haobo %>%
                  group_by(label) %>%
                  sample_n(sum(haobo$label == 0),replace=TRUE) %>%
                  ungroup()) 
  }
  
  ids_test <- haobo_test %>% select(id)

  write_csv(ids_test,file.path(path,'data_out',
                               sprintf('test_set_%s_%s.csv.gz',mod,ss)),
            col_names=TRUE)
  
  
 
  haobo_pre <- haobo_train %>% 
    bind_rows(haobo_test) %>%
    select(where(is.numeric)) %>%
    mutate(across(everything(), ~replace_na(.x, 0))) %>%
    select(where(~sum(if_else(.x > 0,1,0)) >= col_filter)) %>%
    select(id, label,
           starts_with('icd_'),
           starts_with('los'),
           starts_with('count_')) %>%
    select(-icd_sum) %>%
    mutate(label=as.factor(label)) 
  
  ids_train <- haobo_pre %>% anti_join(ids_test,by='id') %>% select(id)
  d_split <- make_splits(x=haobo_pre %>% anti_join(ids_test,by='id') %>% select(-id),
                         assessment=haobo_pre %>% semi_join(ids_test,by='id') %>% select(-id))
  d_test <- testing(d_split)
  d_train <- training(d_split)
  
  
  set.seed(7)
  folds <- vfold_cv(d_train,v=n_folds,strata=label)
  
  tree_spec <- decision_tree(
    cost_complexity = tune(),
    tree_depth = tune(),
    min_n = tune()
  ) %>%
    set_engine("rpart",model=TRUE) %>%
    set_mode("classification")
  
  tuner <- grid_regular(cost_complexity(),
                        tree_depth(c(3,10)),
                        min_n(c(5,75)),
                        levels=n_sweep)
  
  
  
  mets <- metric_set(accuracy, sens, yardstick::spec, f_meas, roc_auc)
  
  wf <- workflow() %>%
    add_model(tree_spec) %>%
    add_formula(label ~ .)
  
  fit <- wf %>%
    tune_grid(resamples=folds,
              grid=tuner,
              control=control_grid(verbose = TRUE),
              metrics=mets)
  
  
  write_rds(list(data=haobo_pre,fit=fit,wf=wf,
                 split=d_split,
                 train_ids=ids_train,test_ids=ids_test),
            file.path(path,'data_in',sprintf('fit_%s_count_del_%s.rds',mod,ss)))


}  
stopCluster(cl)

