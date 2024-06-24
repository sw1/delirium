pacman::p_load(tidymodels,tidyverse,doParallel,ranger,glue,probably,discrim)

# script to perform self training and predict pseudo labels

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
  all_cores <- 4
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
  all_cores <- 14
}
source(file.path(path,'code','fxns.R'))

outfile <- file.path(path,'scratch','semisup_out.txt')
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
fs <- read_rds(file.path(path,'data_out','08_rf_fs.rds')) 

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
  file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds')) %>%
  select(id,set,label,matches(features_subset))

# test set
test <- master %>%
  filter(set == 'test_expert') %>%
  select(-set)

# heldout set
heldout <- master %>%
  filter(set == 'heldout_expert') %>%
  select(-set)

# master without heldout, nrows 153967
master <- master %>%
  anti_join(test,by='id') %>%
  anti_join(heldout,by='id')

# get node size pars to sweep
ns_adapt <- FALSE #c(TRUE,FALSE)
node_size <- params$min_node_perc[1]

# fraction to subset expert labels
fracs <- c(1,0.75,0.5,0.35,0.2)

# create parameter sweep table
# not doing ns adapt anymore since prior runs didnt make a difference
set.seed(342)
combs <- crossing(thresholds,ns_adapt,seeds,fracs) %>% 
  filter(!(fracs < 1 & thresholds != 0.7)) %>%
  group_by(fracs) %>%
  mutate(fracs_seed=sample(1:99999,1)) %>%
  ungroup() %>%
  arrange(desc(fracs),thresholds) %>%
  mutate(b_acc=NA)

cat(glue('Running self-training with {all_cores} ',
         'cores for {nrow(combs)} trials.\n\n'))

t_total_start <- Sys.time()

out <- foreach(i=1:nrow(combs),.combine='c',.verbose=TRUE,
               .errorhandling='stop',
               .export=c('params','tree','all_cores'),
               .packages=c('tidymodels','tidyverse','ranger','glue',
                           'probably','discrim','rsample')) %do% {
                 
                             
                 thres <- combs$thresholds[i]
                 s <- combs$seeds[i]
                 frac <- combs$fracs[i]
                 
                 # labeled set
                 labeled <- master %>%
                   filter(set == 'expert') %>%
                   select(-set)
                 
                 if (frac < 1){
                   set.seed(combs$fracs_seed[i])
                   labeled <- labeled %>%
                     group_by(label) %>%
                     sample_frac(frac) %>%
                     ungroup()
                 } 
                 
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
                 # train <- upsamp(labeled)
                 train <- labeled
                 
                 if (combs$ns_adapt[i]){
                   ns <- get_node_size(node_size,train)
                 }else{
                   ns <- get_node_size(node_size,labeled)
                 }
                 
                 cat(glue('\nRunning self-training ({i}/{nrow(combs)}):\n',
                          '\tseed: {s}\n',
                          '\tfrac: {frac}\n',
                          '\tthreshold: {thres}\n\n
                          '))
                 
                 # fit inital rf on expert labels and calibrate via logistic
                 tree_spec <- rand_forest(
                   mtry  = !!params$mtry[1],
                   trees = !!trees,
                   min_n = !!ns
                 ) %>%
                   set_engine('ranger',
                              replace=TRUE,
                              # num.threads=all_cores,
                              verbose=FALSE,
                              oob.error=FALSE) %>%
                   set_mode('classification')
                 
                 rf <- workflow() %>%
                   add_model(tree_spec) %>%
                   add_formula(label ~ .)
                 
                 train_cv <- vfold_cv(train %>%
                                        mutate(
                                          label=as.factor(if_else(
                                            label == 1,'pos','neg'))),
                                      strata=label,v=10)
                 
                 cal <- rf %>%
                   fit_resamples(train_cv,metrics=metric_set(roc_auc,brier_class),
                                 control=control_resamples(save_pred=TRUE)) %>%
                   cal_estimate_logistic()
                 
                 # predict labels for entire unlabeled dataset to start
                 preds <- rf %>%
                   fit(data=train %>%
                         mutate(label=as.factor(if_else(
                           label == 1,'pos','neg')))) %>%
                   augment(new_data=unlabeled %>%
                             mutate(label=as.factor(if_else(
                               label == 1,'pos','neg')))) %>%
                   cal_apply(cal,pred_class=.pred_class) %>%
                   select(.pred_0=.pred_neg,.pred_1=.pred_pos)

                 
                 rm(list=c('rf','cal','train_cv'))
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
                   # train <- upsamp(labeled)
                   train <- labeled
                   
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
                                num.threads=all_cores, 
                                verbose=FALSE,
                                oob.error=FALSE) %>%
                     set_mode('classification')
                   
                   # fit model with updated training data (predicted labels + expert)
                   rf <- workflow() %>%
                     add_model(tree_spec) %>%
                     add_formula(label ~ .) %>%
                     fit(train %>% select(-id))
                   
                   preds <- predict(rf,unlabeled %>% select(-id),type='prob')
                   
                   # calculate accuracy on test set
                   preds_test <- predict(rf,test %>% select(-id))
                   acc_test <- bal_accuracy(tibble(yhat=preds_test$.pred_class,
                                                   y=test$label),
                                            truth=y,estimate=yhat)$.estimate
                   combs$b_acc[i] <- acc_test
                   
                   t_end <- Sys.time()
                   
                   cat(glue('\n({i}/{nrow(combs)})| ',
                            't: {round(difftime(t_end,t_start,units="mins"),2)}, ',
                            's: {s}, ',
                            'thr: {thres}, ',
                            'frac: {frac}, ',
                            'iter: {r+1}/{n_iter}, ',
                            'tol: {tol_counter}/{tol_stop}, ',
                            'b_acc: {round(acc_test,2)}, ',
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
                           file.path(path,'data_tmp',
                                     glue('labels_rfst_th{thres*100}_seed{s}_',
                                          'frac{frac*100}.csv.gz')))
                 
                 return(list(labeled))
                 
               }

t_total_end <- Sys.time()

stopCluster(cl)

cat(glue('All iterations complete in ',
         '{round(difftime(t_total_end,t_total_start,units="mins"),2)} ',
         'minutes, saving results.\n\n'))

write_rds(list(combs=combs,out=out,time=c(t_total_start,t_total_end)),
          file.path(path,'data_out','10_labels_rfst_count_del_full.rds'))