library(tidymodels)
library(glmnet)
library(tidyverse)
library(doParallel)
library(vip)
library(rpart.plot)
library(icd.data)
library(ranger)
library(glue)

# script to perform initial random forest for self training on expert
# labeled notes only for parameter cross validation

all_cores <- parallel::detectCores(logical=FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

n_sweep <- 4 # number of parameter levels to sweep
n_folds <- 10 # cv folds per parameter level
col_filter <- 50 # filter  features with less than this many sample occurrences
m <- 'f_meas' # metric to optimize

master <- read_rds(file.path(path,'data_in','05_full_icd_tbl.rds')) %>%
  filter(!is.na(label))  # turn off for full data, note heldout removed below

# create icd indicators
features_icds <- unique(unlist(master$icd_codes))
features_icds <- features_icds[nchar(features_icds) > 0]
icd_mat <- matrix(0,nrow(master),length(features_icds))
colnames(icd_mat) <- paste0('icd_',features_icds)
for (i in 1:nrow(master)){
  icds <- na.exclude(unlist(master$icd_codes[i]))
  icds <- icds[nchar(icds) > 0]
  if (length(icds) == 0) next
  icds <- paste0('icd_',icds)
  for (j in seq_along(icds)){
    icd_mat[i,icds[j]] <- icd_mat[i,icds[j]] + 1
  }
}
icd_mat <- icd_mat[,colSums(icd_mat) >= col_filter]

# create service indicators
features_service <- unique(unlist(master$service))
service_mat <- matrix(0,nrow(master),length(features_service))
colnames(service_mat) <- features_service
for (i in 1:nrow(master)){
  servs <- unique(unlist(master$service[i]))
  service_mat[i,servs] <- 1
}
colnames(service_mat) <- paste0('count_service_',colnames(service_mat))
service_mat <- service_mat[,colnames(service_mat) != 'count_service_other']
colnames(service_mat)[
  colnames(service_mat) == 'count_service_obstetrics/gynecology'
] <- 'count_service_ob'
service_mat <- service_mat[,colSums(service_mat) >= col_filter]

# merge feature tables and add raw notes for problem list and hashes
master <- master %>% 
  select(-service,-icd_codes,-icd_codes_del) %>%
  left_join(read_csv(file.path(path,'data_in','notes.csv.gz')) %>%
              select(id=rdr_id,note=note_txt),
            by='id') %>%
  bind_cols(icd_mat) %>%
  bind_cols(service_mat)

# create metadata
master <- create_counts(master,nurse=FALSE)

# select features to be used for rfst
master <- master %>%
  select(id,set,label,los,discharge_date,sex,age,num_meds,num_allergies,
         len_pmhx,term_count_hc,term_count_hpi,
         starts_with('icd_'),starts_with('count_')) %>%
  select(-icd_sum) 

# take out heldout to upsample training data
heldout <- master %>%
  filter(set=='heldout_expert')

master <- master %>%
  anti_join(heldout,by='id')

# upsample smaller class for balanced training
set.seed(12)
master <- upsamp(master)

# rebind heldout set
master <- master %>% 
  bind_rows(heldout) 

# replace missing values with 0 and normalize features
master <- master %>%
  mutate(across(everything(),~replace_na(.x, 0)))
  
cat(glue('\nNumber of features: {ncol(master)-3}.\n\n'))

d_split <- make_splits(x=master %>% 
                         filter(set != 'heldout_expert') %>% 
                         select(-id,-set),
                       assessment=master %>% 
                         filter(set == 'heldout_expert') %>% 
                         select(-id,-set))

write_rds(master,file.path(path,'data_in','06_dat_st_rf.rds'))
rm(master)

d_train <- training(d_split)

# run random forest cv with fixed trees at 1000 and over parameter sweep
set.seed(7)
folds <- vfold_cv(d_train,v=n_folds,strata=label)

tree_spec <- rand_forest(
  mtry  = tune(),
  trees = 1000,
  min_n = tune()
) %>%
  set_engine('ranger',
             regularization.factor=tune('lambda'),
             regularization.usedepth=tune('depth'),
             verbose=TRUE,
             oob.error=FALSE) %>%
  set_mode('classification')

tuner <- grid_regular(mtry(c(5,floor(sqrt(ncol(d_train))))),
                      min_n=min_n(c(5,50)),
                      depth=regularize_depth(),
                      lambda=penalty_L2(),
                      levels=n_sweep)

mets <- metric_set(accuracy,sens,yardstick::spec,f_meas,roc_auc)

wf <- workflow() %>%
  add_model(tree_spec) %>%
  add_formula(label ~ .) 

fit <- wf %>%
  tune_grid(resamples=folds,
            grid=tuner,
            metrics=mets)

# choose best tree within percent loss range
best_tree <- fit %>% 
  select_by_pct_loss(metric=m,limit=5,
                     desc(min_n),desc(mtry),lambda,desc(depth))

# build final model with best parameters with importance measure for
# feature selection
tree_spec <- rand_forest(
  mtry  = best_tree$mtry,
  trees = 1000,
  min_n = best_tree$min_n
) %>%
  set_engine('ranger',
             regularization.factor=best_tree$lambda,
             regularization.usedepth=best_tree$depth,
             importance='impurity',
             verbose=TRUE,
             oob.error=FALSE) %>%
  set_mode('classification')

rf <- workflow() %>%
  add_model(tree_spec) %>%
  add_formula(label ~ .) %>%
  fit(d_train) 

# extract features and save
features <- rf %>% 
  extract_fit_parsnip() %>%
  vip(num_features=ncol(d_train)-1) %>%
  .$data

write_rds(list(fit=fit,wf=wf,split=d_split,features=features),
          file.path(path,'data_in','06_fit_st_rf.rds'))

stopCluster(cl)

