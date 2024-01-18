library(tidymodels)
library(tidyverse)
library(doParallel)
library(ranger)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}

outfile <- file.path(path,'..\\semisup_out.txt')
file.remove(outfile)

all_cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(all_cores,outfile=outfile)
registerDoParallel(cl)

source(file.path(path,'code','fxns.R'))

mod <- 'rf'
subsets <- c('icd','sub_chapter','major')
thresholds <- c(0.95,0.80)
weights <- c(10,5,2,1)
m <- 'f_meas'
tol <- 50
tol_stop <- 2
n_iter <- 10
f_del <- 0.2

id_heldout <- read_csv(file.path(path,'data_out','heldout_tree_set.csv.gz'))

# id_heldout <- read_csv(
#   file.path(path,'to_python','tbl_to_python_updated_treeheldout.csv.gz')) %>%
#   filter(set == 'test_haobo') %>%
#   select(id,label)

id_test <- read_csv(
  file.path(path,'to_python','tbl_to_python_updated.csv.gz')) %>%
  filter(set == 'test_haobo') %>%
  select(id,label) %>%
  anti_join(id_heldout,by='id')

combs <- crossing(thresholds,subsets,weights) %>% 
  arrange(desc(thresholds),desc(weights),subsets) %>%
  mutate(acc=NA)

set.seed(10)
out <- foreach(i=1:nrow(combs),.combine='c',
               .packages=c('tidymodels','tidyverse','ranger')) %dopar% {
  
    ss <- combs$subsets[i]
    thres <- combs$thresholds[i]
    w <- combs$weights[i]
        
    cat(sprintf('\n\nRunning self training for %s, thres %s, w %s.\n\n',
                ss,thres,w))
    
    tree_fit <- read_rds(
      file.path(path,'data_in',
                sprintf('fit_%sreg_count_del_%s.rds',mod,ss)))

    features <- colnames(tree_fit$data)
    
    features_subset <- tree_fit$features %>% 
      slice_head(n=50) %>%
      select(Variable) %>%
      unlist()
    
    haobo_post <- read_rds(
      file.path(path,'data_in',
                sprintf('alldat_preprocessed_for_pred_%s.rds',ss)))
    
    haobo_post <- haobo_post[,features]
    haobo_post <- haobo_post %>%
      mutate(across(everything(), ~replace_na(.x, 0))) %>%
      mutate(label=as.factor(label))
    
    
    haobo_heldout <- haobo_post %>%
      right_join(id_heldout %>% select(id), by='id') %>%
      filter(!is.na(label))
    
    id_heldout <- haobo_heldout %>% select(id,label) 
    
    haobo_post <- haobo_post %>%
      anti_join(haobo_heldout %>% select(id), by='id')
    
    haobo_test <- haobo_post %>%
      right_join(id_test %>% select(id), by='id') %>%
      filter(!is.na(label))
    
    id_test <- haobo_test %>% select(id,label)
    
    haobo_post <- haobo_post %>%
      anti_join(haobo_test %>% select(id), by='id')

    # N_train <- nrow(haobo_post)
    # N_class <- length(unique(na.omit(haobo_post$label)))
    # class_w_inv <- N_train/(c(N_train*(1-f_del),N_train*f_del)*N_class)
    # class_w <- rev(class_w_inv)
    
    best_tree <- tree_fit$fit %>% 
      select_by_pct_loss(metric=m,limit=5,
                         desc(min_n),desc(mtry),lambda,desc(depth))
    
    wf <- tree_fit$wf %>% 
      finalize_workflow(best_tree) %>%
      last_fit(tree_fit$split) %>%
      extract_workflow()
    
    tree_spec <- rand_forest(
      mtry  = best_tree$mtry,
      trees = 1000,
      min_n = best_tree$min_n
    ) %>%
      set_engine("ranger",
                 # regularization.factor=best_tree$lambda,
                 # regularization.usedepth=best_tree$depth,
                 verbose=TRUE,
                 oob.error=FALSE) %>%
      set_mode("classification")
    
    ids <- haobo_post %>% select(id)
    haobo_pred <- haobo_post %>% select(-id)
    
    preds <- wf %>%
      predict(haobo_pred,type='prob')  
    
    set.seed(7)
    r <- 0
    n_pred <- 0
    tol_0 <- 0
    tol_counter <- 0
    while (n_pred < nrow(haobo_post)){
     
      haobo_pred <- preds %>%
        bind_cols(ids) %>%
        mutate(label=case_when(
          .pred_1 >= thres ~ 1,
          .pred_0 >= thres ~ 0,
          TRUE ~ NA
        ),
        label=as.factor(label)) %>%
        select(id,label) %>% 
        left_join(haobo_post %>% select(-label),by='id') %>%
        filter(!is.na(label)) %>%
        bind_rows(haobo_test)
      
      n_pred <- nrow(haobo_pred)
      tol_1 <- n_pred
      tol_diff <- tol_1 - tol_0
      if (tol_diff < tol) tol_counter <- tol_counter + 1
      
      cat(sprintf('subset=%s, threshold=%s, imp=%s, r=%s, counter=%s\ndim predicted: %s\ndim full: %s\n',
                  ss,thres,w,r,tol_counter,nrow(haobo_pred),nrow(haobo_post)))
      
      cat(sprintf('Current labels\n0: %s \n1: %s\n\n',
                  table(haobo_pred$label)[1],table(haobo_pred$label)[2]))
      
      haobo_pred <- haobo_pred %>% 
        mutate(w=importance_weights(if_else(id %in% id_test$id,w,1))) %>%
        select(-id)
      
      # downsampling since upsampling is too memory and time demanding
      # n_downsamp <- min(table(haobo_pred$label))
      # haobo_train <- haobo_pred %>%
      #   group_by(label) %>%
      #   sample_n(n_downsamp,replace=FALSE) %>%
      #   ungroup()
      
      # upsample smaller class
      n_upsamp <- round(sum(haobo_pred$label == 1)/sum(haobo_pred$label == 0))
      haobo_train <- tibble()
      for (i in 1:n_upsamp){
        haobo_train <- haobo_train %>%
          bind_rows(haobo_pred %>%
                      group_by(label) %>%
                      sample_n(sum(haobo_pred$label == 0),replace=TRUE) %>%
                      ungroup())
      }
      
      haobo_train <- haobo_train[,c('label','w',features_subset)]
      
      # haobo_train <- haobo_pred
      
      rf <- workflow() %>%
        add_model(tree_spec) %>%
        add_formula(label ~ .) 
      
      if (w != 1){
        rf <- rf %>%
          add_case_weights(w) %>%
          fit(haobo_train) 
      }else{
        rf <- rf %>%
          fit(haobo_train %>% select(-w))
      }

        # fit(haobo_pred)
      
      # rf <- ranger(label ~ ., data=haobo_train,probability=TRUE,
      #              num.tree=p_trees,mtry=p_mtry,min.node.size=p_min_n,
      #              oob.error=FALSE,verbose=TRUE)
      
      ids <- haobo_post %>% select(id)
      haobo_pred <- haobo_post %>% select(-id)
       
      preds_heldout <- predict(rf,haobo_heldout)
      acc_heldout <- mean(preds_heldout$.pred_class == haobo_heldout$label)
      combs$acc[i] <- acc_heldout
      
      cat(sprintf('\n\nHeldout performance\nsubset=%s, threshold=%s, imp=%s, r=%s, \naccuracy: %s\npredicted 0/1: %s/%s\n',
                  ss,thres,w,r,acc_heldout,table(preds_heldout$.pred_class)[1],table(preds_heldout$.pred_class)[2]))
      
      preds <- predict(rf,haobo_pred,type='prob')
      # preds <- predict(rf,haobo_pred)$predictions
      # colnames(preds) <- c('.pred_0','.pred_1')
    
      r <- r + 1
      
      if (r == n_iter || tol_counter == tol_stop){
  
        cat('\n\nEnding optimizing early per tolerance.\n')
        
        break
      }
      tol_0 <- tol_1
    }
    
    haobo_out <- preds %>%
      bind_cols(ids) %>%
      mutate(label=case_when(
        .pred_1 >= thres ~ 1,
        .pred_0 >= thres ~ 0,
        TRUE ~ NA
      )) %>%
      select(id,label) %>%
      bind_rows(haobo_test %>% select(id,label))
    
    write_csv(haobo_out %>% filter(!is.na(label)),
              file.path(path,'data_in',
                        sprintf('labels_rfst%s_%s_count_del_%s_filt.csv.gz',
                                thres*100,w,ss)),
              col_names=TRUE)
    
    write_csv(haobo_out %>% mutate(label=if_else(is.na(label),0,label)),
              file.path(path,'data_in',
                        sprintf('labels_rfst%s_%s_count_del_%s_full.csv.gz',
                                thres*100,w,ss)),
              col_names=TRUE)
    
    return(list(haobo_out))
    
}

stopCluster(cl)

save_rds(list(combs=combs,out=out), file.path(path,'data_tmp',
                   sprintf('labels_rfst_count_del_full.rds')))
