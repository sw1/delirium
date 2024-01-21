library(tidymodels)
library(tidyverse)
library(doParallel)
library(ranger)
library(glue)

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
thresholds <- c(0.95,0.85,0.70)
weights <- c(5,2,1)
m <- 'f_meas'
n_features <- c(20,50)
tol <- 25
tol_stop <- 2
n_iter <- 10
f_del <- 0.2

id_heldout <- read_csv(file.path(path,'data_out','heldout_tree_set.csv.gz'))

id_test <- read_csv(
  file.path(path,'to_python','tbl_to_python_updated.csv.gz')) %>%
  filter(set == 'test_haobo') %>%
  select(id,label) %>%
  anti_join(id_heldout,by='id')

combs <- crossing(thresholds,subsets,weights,n_features) %>% 
  arrange(desc(thresholds),desc(weights),subsets,desc(n_features)) %>%
  mutate(acc=NA)

set.seed(10)
out <- foreach(i=1:nrow(combs),.combine='c',
               .packages=c('tidymodels','tidyverse','ranger','glue')) %dopar% {
  
    ss <- combs$subsets[i]
    thres <- combs$thresholds[i]
    w <- combs$weights[i]
    f <- combs$n_features[i]
        
    cat(glue('Running self-training ({i}/{nrow(combs)}):\n',
             '\tthreshold:{thres}\n',
             '\tweight: {w}\n',
             '\tsubset: {ss}\n',
             '\tfeatures: {f}\n\n
             '))
    
    tree_fit <- read_rds(
      file.path(path,'data_in',
                sprintf('fit_%sreg_count_del_%s.rds',mod,ss)))

    features <- colnames(tree_fit$data)
    
    features_subset <- tree_fit$features %>% 
      slice_head(n=f) %>%
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
                 verbose=FALSE,
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
    while (TRUE){
     
      haobo_pred <- preds %>%
        bind_cols(ids) %>%
        mutate(label=case_when(
          .pred_1 >= thres ~ 1,
          .pred_0 >= thres ~ 0,
          TRUE ~ NA))
      
      if (sum(is.na(haobo_pred$label)) == 0) {
        
        cat(glue('Status ({i}/{nrow(combs)}): all labels predicted\n',
                 '\tthreshold: {thres}\n',
                 '\tweight: {w}\n',
                 '\tsubset: {ss}\n',
                 '\tfeatures: {f}\n',
                 '\titer: {r}\n',
                 '\tlabels predicted 0/1\\
: {sum(haobo_pred$label == 0,na.rm=TRUE)}/\\
{sum(haobo_pred$label == 1,na.rm=TRUE)}\n\n
                 '))
        
        break
      }
      
      haobo_pred <- haobo_pred %>%
        mutate(label=as.factor(label)) %>%
        select(id,label) %>% 
        left_join(haobo_post %>% select(-label),by='id') %>%
        filter(!is.na(label)) %>%
        bind_rows(haobo_test)
      
      n_pred <- nrow(haobo_pred)
      tol_1 <- n_pred
      tol_diff <- tol_1 - tol_0
      if (tol_diff < tol) tol_counter <- tol_counter + 1
      
      haobo_pred <- haobo_pred %>% 
        mutate(w=importance_weights(if_else(id %in% id_test$id,w,1))) %>%
        select(-id)
      
      n_1 <- sum(haobo_pred$label == 1)
      n_0 <- sum(haobo_pred$label == 0)
      
      # downsampling since upsampling is too memory and time demanding
      # haobo_train <- haobo_pred %>%
      #   group_by(label) %>%
      #   sample_n(min(c(n_1,n_0)),replace=FALSE) %>%
      #   ungroup()
      
      # upsample smaller class
      n_upsamp_max <- max(c(n_0,n_1))
      n_upsamp_min <- min(c(n_0,n_1))
      n_upsamp <- round(n_upsamp_max/n_upsamp_min)
      haobo_train <- tibble()
      for (j in 1:n_upsamp){
        haobo_train <- haobo_train %>%
          bind_rows(haobo_pred %>%
                      group_by(label) %>%
                      sample_n(n_upsamp_min,replace=TRUE) %>%
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
      
      cat(glue('Status ({i}/{nrow(combs)}):\n',
               '\tthreshold:{thres}\n',
               '\tweight: {w}\n',
               '\tsubset: {ss}\n',
               '\tfeatures: {f}\n',
               '\titer: {r}/{n_iter}\n',
               '\tcounter: {tol_counter}/{tol_stop}\n',
               '\tdim predicted/full: {n_pred}/{nrow(haobo_post)}\n',
               '\tlabels predicted 0/1: {n_0}/{n_1}\n',
               '\taccuracy: {acc_heldout}\n',
               '\tpredicted 0/1: {table(preds_heldout$.pred_class)[1]}',
               '/{table(preds_heldout$.pred_class)[2]}\n\n
               '))
      
      preds <- predict(rf,haobo_pred,type='prob')
      # preds <- predict(rf,haobo_pred)$predictions
      # colnames(preds) <- c('.pred_0','.pred_1')
    
      r <- r + 1
      
      if (r == n_iter | tol_counter == tol_stop){
        
        cat(glue('Status ({i}/{nrow(combs)}): ending optimizing early per t\\
olerance\n',
                 '\tthreshold:{thres}\n',
                 '\tweight: {w}\n',
                 '\tsubset: {ss}\n',
                 '\tfeatures: {f}\n',
                 '\titer: {r}\n',
                 '\tcounter: {tol_counter}\n\n
                 '))
        
        cat('Generating final predictions\n')
        
        haobo_pred <- preds %>%
          bind_cols(ids) %>%
          mutate(label=case_when(
            .pred_1 >= thres ~ 1,
            .pred_0 >= thres ~ 0,
            TRUE ~ NA)) 
        
        break
      }
      tol_0 <- tol_1
    }
    
    haobo_out <- haobo_pred %>%
      select(id,label) %>%
      mutate(label=as.factor(label)) %>%
      bind_rows(haobo_test %>% select(id,label))
    
    cat('Saving restults\n\n')
    
    # do the filter/threshold is next script
    write_csv(haobo_out,
              file.path(path,'data_in',
                        sprintf(
                          'labels_rfst_th%s_w%s_f%s_count_del_%s.csv.gz',
                                thres*100,w,ss,f)),
              col_names=TRUE)
    
    return(list(haobo_out))
    
}

stopCluster(cl)

write_rds(list(combs=combs,out=out), file.path(path,'data_tmp',
                   sprintf('labels_rfst_count_del_full.rds')))
