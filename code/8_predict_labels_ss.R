library(tidymodels)
library(glmnet)
library(tidyverse)
library(doParallel)
library(vip)
library(rpart.plot)
library(ranger)

all_cores <- parallel::detectCores(logical = TRUE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

mod <- 'rf'
subsets <- c('icd','sub_chapter','major') 

m <- 'f_meas'
thresholds <- c(0.7,0.6)
tol <- 10
tol_stop <- 2
n_iter <- 10

id_haobo <- read_csv(
  file.path(path,'to_python','tbl_to_python_updated.csv.gz')) %>%
  filter(set == 'test_haobo') %>%
  select(id,label)

for (ss in subsets){
  for (thres in thresholds){
    
    cat(sprintf('\n\nRunning self training for %s %s.\n\n',ss,thres))
    
    tree_fit <- read_rds(
      file.path(path,'data_in',
                sprintf('fit_%s_count_del_%s.rds',mod,ss)))
    
    features <- tree_fit$data %>% colnames()
    
    haobo_post <- read_rds(
      file.path(path,'data_in',
                sprintf('alldat_preprocessed_for_pred_%s.rds',ss))) %>%
      select(id,label,contains(features)) %>%
      mutate(across(everything(), ~replace_na(.x, 0))) %>%
      mutate(label=as.factor(label))
    haobo_post <- haobo_post[,features]
    
    best_tree <- tree_fit$fit %>% 
      select_by_pct_loss(metric=m,limit=5,desc(min_n),trees,desc(mtry))
    
    p_trees <- best_tree$trees # set to 50 for now for speed 
    p_min_n <- best_tree$min_n
    p_mtry <- best_tree$mtry
    
    wf <- tree_fit$wf %>% 
      finalize_workflow(best_tree) %>%
      last_fit(tree_fit$split) %>%
      extract_workflow()
    
    ids <- haobo_post %>% select(id)
    haobo_pred <- haobo_post %>% select(-id)
    
    preds <- wf %>%
      predict(haobo_pred,type='prob')  
      
    tree_spec <- rand_forest(
      mtry  = p_mtry,
      trees = p_trees,
      min_n = p_min_n
    ) %>%
      set_engine("ranger",num.threads=all_cores,verbose=TRUE) %>%
      set_mode("classification")
    
    set.seed(7)
    r <- 0
    n_pred <- 0
    tol_0 <- 99999
    tol_counter <- 0
    while (n_pred < nrow(haobo_post)){
     
      haobo_pred <- preds %>%
        bind_cols(ids) %>%
        left_join(id_haobo,by='id') %>%
        mutate(label=case_when(
          !is.na(label) ~ label,
          .pred_1 >= thres ~ 1,
          .pred_0 >= thres ~ 0,
          TRUE ~ NA
        )) %>%
        select(id,label) %>% 
        left_join(haobo_post %>% select(-label),by='id') %>%
        filter(!is.na(label))
      
      n_pred <- nrow(haobo_pred)
      tol_1 <- n_pred
      tol_diff <- tol_0 - tol_1
      if (tol_diff < tol) tol_counter <- tol_counter + 1
      
      cat(sprintf('\n\nDim pred %s, dim full %s, r=%s.\n\n',
                  nrow(haobo_pred),nrow(haobo_post),r))
      
      cat(sprintf('\n\nCurrent labels\n0: %s \n1: %s\n\n',
                  table(haobo_pred$label)[1],table(haobo_pred$label)[2]))
      
      ids <- haobo_pred %>% select(id)
      haobo_pred <- haobo_pred %>% select(-id) %>% 
        mutate(label=as.factor(label))
      
      rf <- tree_spec %>%
        fit(label ~ ., data = haobo_pred)
      
      ids <- haobo_post %>% select(id)
      haobo_pred <- haobo_post %>% select(-id)
       
      preds <- predict(rf,haobo_pred,type='prob')
    
      r <- r + 1
      
      if (r == n_iter || tol_counter == tol_stop){
        haobo_pred <- preds %>%
          bind_cols(ids) %>%
          left_join(id_haobo,by='id') %>%
          mutate(label=case_when(
            !is.na(label) ~ label,
            .pred_1 >= thres ~ 1,
            TRUE ~ 0
          )) %>%
          select(id,label)
        
        cat(sprintf('\n\nFinal labels\n0: %s \n1: %s\n\n',
                    table(haobo_pred$label)[1],table(haobo_pred$label)[2]))
        
        break
      }
      tol_0 <- tol_1
    }
      
      
    write_csv(haobo_pred,file.path(path,'data_in',
                               sprintf('labels_rfst%s_count_del_%s.csv.gz',
                                       thres*100,ss)),
              col_names=TRUE)
  }
}

stopCluster(cl)
