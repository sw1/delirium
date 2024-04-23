library(tidymodels)
library(tidyverse)
library(doParallel)
library(ranger)
library(glue)

# script to perform self training and predict pseudo labels

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

outfile <- file.path(path,'..\\semisup_out.txt')
file.remove(outfile)

all_cores <- parallel::detectCores(logical=FALSE)
cl <- makePSOCKcluster(all_cores,outfile=outfile)
registerDoParallel(cl)

m <- 'f_meas'
thresholds <- c(0.80,0.7) # thresholds for pseudo label
n_features <- c(50,125) # number of features to use
seeds <- c(1834,4532,8010) # random seeds
tol <- 50 # tolerance for stopping early
tol_stop <- 2 # number of times to hit tolerance before stopping
n_iter <- 10 # max number of iterations

# create parameter sweep table
combs <- crossing(thresholds,n_features,seeds) %>% 
  arrange(thresholds,n_features) %>%
  mutate(acc=NA)

cat(glue('Running self-training with {all_cores} ',
         'cores for {nrow(combs)} trials.\n\n'))

out <- foreach(i=1:nrow(combs),.combine='c',
               .packages=c('tidymodels','tidyverse','ranger','glue')) %dopar% {
  
    thres <- combs$thresholds[i]
    f <- combs$n_features[i]
    s <- combs$seeds[i]
    
    set.seed(s)
    
    cat(glue('\nRunning self-training ({i}/{nrow(combs)}):\n',
             '\tseed: {s}\n',
             '\tthreshold: {thres}\n',
             '\tfeatures: {f}\n\n
             '))
    
    tree_fit <- read_rds(
      file.path(path,'data_in','06_fit_st_rf.rds'))

    # full feature set that was used for initial training
    # features <- training(tree_fit$split) %>%
    #   select(-label) %>%
    #   colnames() %>%
    #   paste('^',.,'$',sep='')
    
    # subset of features after subset selection
    features_subset <- tree_fit$features %>% 
      slice_head(n=f) %>%
      pull(Variable) %>%
      paste('^',.,'$',sep='')
    
    # load dataset and filter features
    master <- read_rds(
      file.path(path,'data_in','07_alldat_preprocessed_for_pred.rds')) %>%
      select(id,set,label,matches(features_subset))
    
    # heldout set
    heldout <- master %>%
      filter(set == 'heldout_expert') %>%
      select(-set)
    
    # labeled set
    labeled <- master %>%
      filter(set == 'test_expert') %>%
      select(-set)
    
    # unlabeled set
    unlabeled <- master %>%
      select(-set) %>%
      anti_join(heldout,by='id') %>%
      anti_join(labeled,by='id')
    
    # best tree and workflow from cv based on metric (f1)
    best_tree <- tree_fit$fit %>% 
      select_by_pct_loss(metric=m,limit=5,
                         desc(min_n),desc(mtry),lambda,desc(depth))
    
    # build rf model with final parameters from cv
    tree_spec <- rand_forest(
      mtry  = best_tree$mtry,
      trees = 1000,
      min_n = best_tree$min_n
    ) %>%
      set_engine('ranger',
                 verbose=FALSE,
                 oob.error=FALSE) %>%
      set_mode('classification')
    
    rf <- workflow() %>%
      add_model(tree_spec) %>%
      add_formula(label ~ .) %>%
      fit(labeled %>% select(-id))
    
    # predict labels for entire unlabeled dataset to start
    preds <- predict(rf,unlabeled %>% select(-id),type='prob')
    
    rm(list=c('master','tree_fit','rf'))
    gc()
    
    r <- 0
    n_pred <- 0
    tol_0 <- 0
    tol_counter <- 0
    while (TRUE){
     
      # rebind ids, set labels to 1,0 based on threshold, filter unlabeled
      preds <- preds %>%
        bind_cols(unlabeled %>% select(id)) %>%
        mutate(label=case_when(
          .pred_1 >= thres ~ 1,
          .pred_0 >= thres ~ 0,
          TRUE ~ NA)) %>%
        filter(!is.na(label)) %>%
        mutate(label=as.factor(label)) %>%
        select(id,label) 
      
      # update labeled dataset with newly labeled examples
      labeled <- labeled %>%
        bind_rows(unlabeled %>%
                    select(-label) %>%
                    right_join(preds,by='id'))
      
      # remove newly labeled data from unlabeled dataset
      unlabeled <- unlabeled %>%
        anti_join(preds %>% select(id),by='id')
      
      # end loop if all examples are labeled
      if (nrow(unlabeled) == 0) {
        
        cat(glue('\nStatus ({i}/{nrow(combs)}): all labels predicted\n',
                 '\tseed: {s}\n',
                 '\tthreshold: {thres}\n',
                 '\tfeatures: {f}\n',
                 '\titer: {r}\n',
                 '\tlabels predicted 0/1: ',
                 '{sum(labeled$label == 0,na.rm=TRUE)}/',
                 '{sum(labeled$label == 1,na.rm=TRUE)}\n\n'))
        
        break
        
      }
      
      n_pred <- nrow(labeled)
      tol_1 <- n_pred
      tol_diff <- tol_1 - tol_0
      if (tol_diff < tol) tol_counter <- tol_counter + 1
      
      # upsample smaller class for training
      train <- upsamp(labeled)
      
      # fit model with updated training data (predicted labels + expert)
      rf <- workflow() %>%
        add_model(tree_spec) %>%
        add_formula(label ~ .) %>%
        fit(train %>% select(-id))
      
      preds <- predict(rf,unlabeled %>% select(-id),type='prob')

      # calculate accuracy on heldout set
      preds_heldout <- predict(rf,heldout %>% select(-id))
      acc_heldout <- mean(preds_heldout$.pred_class == heldout$label)
      combs$acc[i] <- acc_heldout
      
      cat(glue('\nStatus ({i}/{nrow(combs)}):\n',
               '\tseed: {s}\n',
               '\tthreshold:{thres}\n',
               '\tfeatures: {f}\n',
               '\titer: {r}/{n_iter}\n',
               '\tcounter: {tol_counter}/{tol_stop}\n',
               '\taccuracy: {round(acc_heldout,2)}\n',
               '\ttraining samples per class after upsamp: {nrow(train)/2}\n',
               '\tpredicted 0/1: {table(preds_heldout$.pred_class)[1]}',
               '/{table(preds_heldout$.pred_class)[2]}\n',
               '\tcurrent samples labeled 0/1 (lab/unlab): ',
               '{sum(labeled$label == 0,na.rm=TRUE)}',
               '/{sum(labeled$label == 1,na.rm=TRUE)} ',
               '({nrow(labeled)}/{nrow(unlabeled)})\n\n'))
      
      gc()
      
      r <- r + 1
      
      # if hit max iterations or tolerance, end early
      if (r == n_iter | tol_counter == tol_stop){
        
        cat(glue('\nStatus ({i}/{nrow(combs)}): ending optimization',
                 'early per tolerance\n',
                 '\tseed: {s}\n',
                 '\tthreshold:{thres}\n',
                 '\titer: {r}\n',
                 '\tcounter: {tol_counter}\n\n',
                 'Generating final predictions\n\n
                 '))

        preds <- preds %>%
          bind_cols(unlabeled %>% select(id)) %>%
          mutate(label=case_when(
            .pred_1 >= thres ~ 1,
            .pred_0 >= thres ~ 0,
            TRUE ~ NA)) %>%
          filter(!is.na(label)) %>%
          mutate(label=as.factor(label)) %>%
          select(id,label) 
        
        # final update of labeled dataset with newly labeled examples
        labeled <- labeled %>%
          bind_rows(unlabeled %>%
                      select(-label) %>%
                      right_join(preds,by='id'))
        
        break
        
      }
      
      tol_0 <- tol_1
      
    }
    
    labeled <- labeled %>% 
      select(id,label)
    
    cat(glue('Status ({i}/{nrow(combs)}): saving results'))
    
    write_csv(labeled,
              file.path(
                path,
                'data_in',
                glue('labels_rfst_th{thres*100}_nfeat{f}_seed{s}.csv.gz')))
    
    return(list(labeled))
    
}

stopCluster(cl)

cat(glue('All iterations complete: saving results'))

write_rds(list(combs=combs,out=out),
          file.path(path,'data_tmp','08_labels_rfst_count_del_full.rds'))
