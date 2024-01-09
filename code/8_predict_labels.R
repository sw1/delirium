library(tidymodels)
library(glmnet)
library(tidyverse)
library(doParallel)
library(vip)
library(rpart.plot)
library(ranger)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

mods <- c('rf','tree')
subsets <- c('icd','sub_chapter','major')

ds <- 'count_del'

m <- 'f_meas'

for (mod in mods){
  for (ss in subsets){
    
    haobo_post <- read_rds(file.path(path,
                                     'data_in',
                                     sprintf('alldat_preprocessed_for_pred_%s.rds',ss)))
    
    # change this to have rf and tree
    tree_fit <- read_rds(file.path(path,'data_in',sprintf('fit_%s_count_del_%s.rds',mod,ss)))
    
    if (mod == 'rf'){
      best_tree <- tree_fit$fit %>% select_by_pct_loss(metric=m,limit=5,desc(min_n),trees,desc(mtry))
    }
    if (mod == 'tree'){
      best_tree <- tree_fit$fit %>% select_by_pct_loss(metric=m,limit=5,desc(min_n),tree_depth)
    }
    
    wf <- tree_fit$wf %>% 
      finalize_workflow(best_tree) %>%
      last_fit(tree_fit$split) %>%
      extract_workflow()
    
    features <- tree_fit$data %>% select(-id,-label) %>% colnames()
    
    haobo_pred <- haobo_post %>%
      select(id,label,contains(features)) %>%
      mutate(across(everything(), ~replace_na(.x, 0))) %>%
      mutate(label=as.factor(label))
    
    ids <- haobo_pred %>% select(id)
    haobo_pred <- haobo_pred %>% select(-id)
    
    if (mod == 'rf'){
      preds <- wf %>%
        predict(haobo_pred,type='prob') %>%
        select(label_tree=.pred_1)
      labels <- ids %>% bind_cols(preds) 
    }
    if (mod == 'tree'){
      preds <- wf %>%
        predict(haobo_pred)
      labels <- ids %>% bind_cols(preds) %>% rename(label_tree=.pred_class)
    }

    write_csv(labels,file.path(path,'data_in',sprintf('labels_%s_count_del_%s.csv.gz',mod,ss)),
              col_names=TRUE)
    
  }
}