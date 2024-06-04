pacman::p_load(tidymodels,tidyverse,doParallel,ranger,glue)

# script to perform self training and predict pseudo labels

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
  all_cores <- 4
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
<<<<<<< HEAD
  all_cores <- 8
=======
>>>>>>> 67e45b50c18088b99e1954a4b30d6b48e46b1fc5
}
source(file.path(path,'code','fxns.R'))

outfile <- file.path(path,'..\\semisup_out.txt')
if (file.exists(outfile)) file.remove(outfile)

#all_cores <- parallel::detectCores(logical=FALSE)
cl <- makePSOCKcluster(all_cores,outfile=outfile)
registerDoParallel(cl)

testing <- FALSE
thresholds <- c(0.6,0.7,0.8,0.9) # thresholds for pseudo label
seeds <- c(1834,4532,8010) # random seeds
tol <- 50 # tolerance for stopping early
tol_stop <- 5 # number of times to hit tolerance before stopping
n_iter <- 25 # max number of iterations
trees <- 1500 # number of trees for rf

# get params from feature select and vc sweep
# get feature selection results
fs <- read_rds(file.path(path,'data_in','08_rf_fs.rds')) 

# select best model with reduced features 
# within 1 sd of best performing model based on rmse
params <- fs$perf %>%
  filter(rmse < min(rmse) + sd(rmse),
         n_feats < 200) %>%
  arrange(rmse) %>%
  slice_head(n=1)

features_subset <- fs$features %>%
  arrange(desc(Importance)) %>%
  slice_head(n=params$n_feats[1]) %>%
  pull(Variable) %>%
  paste('^',.,'$',sep='')

# load dataset and filter features, nrows 154267
master <- read_rds(
  file.path(path,'data_in','09_alldat_preprocessed_for_pred.rds')) %>%
  select(id,set,label,matches(features_subset))

# heldout set
heldout <- master %>%
  filter(set == 'heldout_expert') %>%
  select(-set)

# master without heldout, nrows 153967
master <- master %>%
  anti_join(heldout,by='id')

# get node size pars to sweep
ns_adapt <- FALSE #c(TRUE,FALSE)
node_size <- params$min_node_perc[1]

# create parameter sweep table
combs <- crossing(thresholds,ns_adapt,seeds) %>% 
  arrange(thresholds,ns_adapt) %>%
  mutate(acc=NA)

cat(glue('Running self-training with {all_cores} ',
         'cores for {nrow(combs)} trials.\n\n'))

t_total_start <- Sys.time()

out <- foreach(i=1:nrow(combs),.combine='c',.verbose=TRUE,
               .errorhandling='pass',
               .packages=c('tidymodels','tidyverse','ranger','glue')) %dopar% {
  
    thres <- combs$thresholds[i]
    s <- combs$seeds[i]
    
    # labeled set
    labeled <- master %>%
      filter(set == 'test_expert') %>%
      select(-set)
    
    # unlabeled set
    unlabeled <- master %>%
      select(-set) %>%
      anti_join(labeled,by='id')
    
    if (testing){
      cat(glue('\nnrow master: {nrow(master)}\t',
               'nrow sum {nrow(labeled) + nrow(unlabeled)}\n',
               'nrow labeled: {nrow(labeled)}\t',
               'nrow unlabeled: {nrow(unlabeled)}\n\n'))
    }
          
    set.seed(s)
    
    # upsample minority class
    train <- upsamp(labeled)
    
    if (combs$ns_adapt[i]){
      ns <- get_node_size(node_size,train)
    }else{
      ns <- get_node_size(node_size,labeled)
    }
    
    cat(glue('\nRunning self-training ({i}/{nrow(combs)}):\n',
             '\tseed: {s}\n',
             '\tnode size: {ns} ({as.integer(combs$ns_adapt[i])})\n',
             '\tthreshold: {thres}\n\n
             '))
    
    # build rf model with final parameters from tuning
    # note that min_n will adapt based on size of training data
    # to hopefully constrains tree depth
    tree_spec <- rand_forest(
      mtry  = params$mtry[1],
      trees = trees,
      min_n = ns
    ) %>%
      set_engine('ranger',
                 replace=TRUE, 
                 num.threads=1, # testing this
                 verbose=FALSE,
                 oob.error=FALSE) %>%
      set_mode('classification')
    
    rf <- workflow() %>%
      add_model(tree_spec) %>%
      add_formula(label ~ .) %>%
      fit(train %>% select(-id))
    
    # predict labels for entire unlabeled dataset to start
    preds <- predict(rf,unlabeled %>% select(-id),type='prob')
    
    rm(list=c('rf'))
    gc()
    
    r <- 0
    n_pred <- 0
    tol_0 <- 0
    tol_counter <- 0
    while (TRUE){
      
      if (testing){
        cat(glue('\nnrow master: {nrow(master)}\t',
                 'nrow sum {nrow(labeled) + nrow(unlabeled)}\n',
                 'nrow labeled: {nrow(labeled)}\t',
                 'nrow unlabeled: {nrow(unlabeled)}\n\n'))
      }
     
      t_start <- Sys.time()
      
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
      
      n_updated <- nrow(preds)
      
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
        
        if (testing){
          cat(glue('\nnrow master: {nrow(master)}\t',
                   'nrow sum {nrow(labeled) + nrow(unlabeled)}\n',
                   'nrow labeled: {nrow(labeled)}\t',
                   'nrow unlabeled: {nrow(unlabeled)}\n\n'))
        }
        
        cat(glue('\nStatus ({i}/{nrow(combs)}): all labels predicted\n',
                 '\tseed: {s}\n',
                 '\tthreshold: {thres}\n',
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
      
      # get updated node size param based on param
      if (combs$ns_adapt[i]) ns <- get_node_size(node_size,train)
    
      # redefine treespec to adapt to more samples for min_n
      # otherwise, tree depth will increase with more data
      tree_spec <- rand_forest(
        mtry  = params$mtry[1],
        trees = trees,
        min_n = ns
      ) %>%
        set_engine('ranger',
                   replace=TRUE,
                   num.threads=1, # testing
                   verbose=FALSE,
                   oob.error=FALSE) %>%
        set_mode('classification')
      
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
      
      t_end <- Sys.time()
      
      cat(glue('\n({i}/{nrow(combs)})| ',
               't: {round(difftime(t_end,t_start,units="mins"),2)}, ',
               's: {s}, ',
               'thr: {thres}, ',
               'ns: {ns} ({as.integer(combs$ns_adapt[i])}), ',
               'iter: {r+1}/{n_iter}, ',
               'tol: {tol_counter}/{tol_stop}, ',
               'acc: {round(acc_heldout,2)}, ',
               'update: {n_updated}, ',
               'lab 0/1 (unlab): ',
               '{sum(labeled$label == 0,na.rm=TRUE)}',
               '/{sum(labeled$label == 1,na.rm=TRUE)} ',
               '({nrow(unlabeled)}), ',
               'nrow master: {nrow(master)}, ',
               'nrow sum: {nrow(labeled) + nrow(unlabeled)},',
               'nrow labeled: {nrow(labeled)}, ',
               'nrow unlabeled: {nrow(unlabeled)}\n\n'))
      
      gc()
      
      r <- r + 1
      
      # if hit max iterations or tolerance, end early
      if (r == n_iter | tol_counter == tol_stop){
        
        cat(glue('\nStatus ({i}/{nrow(combs)}): ending optimization ',
                 'early per tolerance\n\n
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
        
        # remove newly labeled data from unlabeled dataset
        unlabeled <- unlabeled %>%
          anti_join(preds %>% select(id),by='id')
        
        labeled <- labeled %>%
          bind_rows(unlabeled) 
        
        break
        
      }
      
      tol_0 <- tol_1
      
    }
    
    if (testing){
      cat(glue('\nnrow master: {nrow(master)}\t',
               'nrow sum {nrow(labeled) + nrow(unlabeled)}\n',
               'nrow labeled: {nrow(labeled)}\t',
               'nrow unlabeled: {nrow(unlabeled)}\n\n'))
    }
    
    labeled <- labeled %>% 
      select(id,label)
    
    cat(glue('Status ({i}/{nrow(combs)}): saving results\n\n'))
    
    write_csv(labeled,
              file.path(path,'data_in',
                glue('labels_rfst_th{thres}_seed{s}_',
                     'ns{as.integer(combs$ns_adapt[i])}.csv.gz')))
    
    return(list(labeled))
    
}

t_total_end <- Sys.time()

stopCluster(cl)

cat(glue('All iterations complete in ',
         '{round(difftime(t_total_end,t_total_start,units="mins"),2)} ',
         'minutes, saving results.\n\n'))

write_rds(list(combs=combs,out=out,time=c(t_total_start,t_total_end)),
          file.path(path,'data_tmp','10_labels_rfst_count_del_full.rds'))
