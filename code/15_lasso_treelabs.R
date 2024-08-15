pacman::p_load(text2vec,stopwords,glmnet,tidymodels,
               tidyverse,doParallel,caret,glue,probably)

# script to perform lasso on doc embeddings

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
  all_cores <- 4
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
  all_cores <- 10
}
source(file.path(path,'code','fxns.R'))

#all_cores <- parallel::detectCores(logical=FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

# params
s <- 123 # seed for lasso
nfolds <- 10

train <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  select(-hpi,-hc,-hpi_hc)
test <- train %>%
  filter(set == 'heldout_expert')
train <- train %>%
  filter(set == 'train')

dv_train <- read_rds(file.path(path,'data_out','14_doc_vectors_train.rds')) 
dv_test <- read_rds(file.path(path,'data_out',
                              '14_doc_vectors_heldout_expert.rds')) 

ids_train <- train %>% pull(id) %>% unique()
ids_test <- test %>% pull(id) %>% unique()

dv_train <- dv_train[rownames(dv_train) %in% ids_train,]
dv_test <- dv_test[rownames(dv_test) %in% ids_test,]

y_test <- test %>% 
  select(id,y=label) %>%
  filter(y != -1) 

x_test <- dv_test[as.character(y_test$id),]
colnames(x_test) <- paste0('f',1:ncol(x_test))

x_test <- as_tibble(x_test) %>%
  bind_cols(y=as.factor(unname(y_test$y)))

labs <- train %>% select(starts_with('label')) %>% names()

tbl_res <- tibble(labs) %>%
  rename(fn=labs) %>%
  separate(fn,c('fit','label','threshold','fraction'),
           sep='_',fill='right',remove=FALSE) %>%
  select(-fit) %>%
  mutate(label=if_else(is.na(label),'onlyexpert',label),
         threshold=as.integer(str_extract(threshold,'[0-9]+')),
         fraction=as.integer(str_extract(fraction,'[0-9]+'))) %>%
  mutate(lambda=0,b_acc=0,prec=0,rec=0,f1=0,prop1=0) %>%
  crossing(w=c(-30,-20,-10,-5,0,5,10,20,30))

for (i in 1:nrow(tbl_res)){
  
  cat(glue('\n({i}) fitting lasso for {tbl_res$fn[i]} ',
           'with w={tbl_res$w[i]}\n\n',.na=NA))
  
  y_train <- train %>% 
    select(id,y=all_of(tbl_res$fn[i])) %>%
    filter(y != -1)
  
  w <- if (tbl_res$w[i] >= 0) c(abs(tbl_res$w[i]),1) else c(1,abs(tbl_res$w[i]))
  
  x_train <- dv_train[as.character(y_train$id),]
  colnames(x_train) <- paste0('f',1:ncol(x_train))
  
  x_train <- as_tibble(x_train) %>%
    bind_cols(y=as.factor(unname(y_train$y))) 
  
  if (tbl_res$w[i] != 0) x_train <- x_train %>%
    mutate(w=if_else(y == 1,w[2],w[1]),
           w=importance_weights(w)) 
  
  set.seed(s)
  
  folds <- vfold_cv(x_train,strata=y,v=all_cores)
  
  mod <- logistic_reg(penalty = tune(),mixture = 1) %>%
    set_engine('glmnet') %>%
    set_mode('classification')
  
  rec <- recipe(y ~ .,data=x_train) %>%
    step_normalize(all_numeric_predictors())
  
  wf <- workflow() %>%
    add_model(mod) %>%
    add_recipe(rec) 
  
  if (tbl_res$w[i] != 0) wf <- wf %>% add_case_weights(w) 
  
  mets <- metric_set(yardstick::sensitivity, yardstick::specificity,
                     roc_auc,bal_accuracy)
  
  grid <- tibble(penalty = 10^seq(-3, 0, length.out = 5))
  
  res <- wf %>%
    tune_grid(resamples=folds,grid=grid,metrics=mets) 
  
  # autoplot(res)
  # 
  # uwf <- wf %>%
  #   remove_case_weights()
  # 
  # ures <- uwf %>%
  #   tune_grid(resamples=folds,grid=grid,metrics=mets)
  # 
  # autoplot(ures)

  lambda <- res %>% 
    select_best(metric='bal_accuracy') %>%
    pull(penalty)
  
  mod <- logistic_reg(penalty=!!lambda,mixture = 0.5) %>%
    set_engine('glmnet') %>%
    set_mode('classification')
  
  wf <- workflow() %>%
    add_formula(y ~ .) %>%
    add_model(mod) 
  
  if (tbl_res$w[i] != 0) wf <- wf %>% add_case_weights(w) 
    
  y_hat <- wf %>%
    fit(data=x_train %>%
                   mutate(y=as.factor(if_else(y == 1,'pos','neg')))) %>%
    augment(new_data=x_test %>%
              mutate(y=as.factor(if_else(y == 1,'pos','neg'))))
  
  cal <- try({
    wf %>%
      fit_resamples(vfold_cv(x_train %>%
                               mutate(y=as.factor(if_else(
                                 y == 1,'pos','neg'))),
                             strata=y,v=all_cores),
                    metrics=metric_set(roc_auc,brier_class),
                    control=control_resamples(save_pred=TRUE)) %>%
      cal_estimate_logistic()
  },silent=TRUE)
  
  if (inherits(cal, 'try-error')){
    y_hat <- y_hat %>%
      select(pred_1=.pred_pos)
  }else{
    y_hat <- y_hat %>%
      cal_apply(cal,pred_class=.pred_class) %>%
      select(pred_1=.pred_pos)
  }

  tbl_yhat <- tibble(id=y_test$id,
                     pred=if_else(y_hat[,1] > 0.5,1,0)) %>%
    mutate(pred=factor(pred,levels=0:1)) %>%
    left_join(y_test %>% mutate(y=factor(y,level=0:1)),by='id')
  
  conf <- confusionMatrix(table(tbl_yhat$y,tbl_yhat$pred),
                          mode='everything',
                          positive='1')
  
  perf <- c(conf$byClass[11],
            conf$byClass[5],
            conf$byClass[6],
            conf$byClass[7],
            table(tbl_yhat$pred)[2]/length(tbl_yhat$pred))
  names(perf) <- c('b_acc','prec','rec','f1','prop1')
  
  tbl_res <- tbl_res %>%
    rows_update(as_tibble_row(perf) %>% 
                  mutate(lambda=lambda,
                         fn=tbl_res$fn[i],
                         w=tbl_res$w[i]),
                by=c('fn','w'))
  
  print(tbl_res %>% select(-fn),n=i)
  
}

write_csv(tbl_res,file.path(path,'res','15_lasso_results.csv.gz'))

stopCluster(cl)
