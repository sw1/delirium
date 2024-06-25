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

res <- tibble(labs) %>%
  rename(fn=labs) %>%
  separate(fn,c('fit','label','threshold','fraction'),
           sep='_',fill='right',remove=FALSE) %>%
  select(-fit) %>%
  mutate(label=if_else(is.na(label),'onlyexpert',label),
         threshold=as.integer(str_extract(threshold,'[0-9]+')),
         fraction=as.integer(str_extract(fraction,'[0-9]+'))) %>%
  mutate(lambda=0,b_acc=0,prec=0,rec=0,f1=0,prop1=0)

for (i in seq_along(labs)){
  
  lab <- labs[i]
  
  cat(glue('\n({i}) fitting lasso for {lab}\n\n',.na=NA))
  
  y_train <- train %>% 
    select(id,y=all_of(lab)) %>%
    filter(y != -1)
  
  # calculate weights for class imbalance
  w <- nrow(y_train)/(table(y_train$y) * length(unique(y_train$y)))
  #w <- 1/(table(y_train$y)/nrow(y_train))
  w <- w/sum(w)
  w <- w[y_train$y+1]
  
  x_train <- dv_train[as.character(y_train$id),]
  colnames(x_train) <- paste0('f',1:ncol(x_train))
  
  x_train <- as_tibble(x_train) %>%
    #bind_cols(tibble(w=as.numeric(w))) %>%
    #mutate(w=importance_weights(w)) %>%
    bind_cols(y=as.factor(unname(y_train$y)))
    
  set.seed(s)
  
  mod <- logistic_reg(penalty = tune(),mixture = 0.5) %>%
    set_engine('glmnet') %>%
    set_mode('classification')
    
  lambda <- workflow() %>%
    add_model(mod) %>%
    #add_recipe(y ~ .,data=x_train) %>%
    add_formula(y ~ .) %>%
    #add_case_weights(w) %>%
    tune_grid(resamples=vfold_cv(x_train,strata=y,v=all_cores),
              grid=5,
              metrics=metric_set(bal_accuracy)) %>% 
    select_best(metric='bal_accuracy') %>%
    pull(penalty)
  
  mod <- logistic_reg(penalty=!!lambda,mixture = 0.5) %>%
    set_engine('glmnet') %>%
    set_mode('classification')
  
  wf <- workflow() %>%
    add_formula(y ~ .) %>%
    #add_case_weights(w) %>%
    add_model(mod) 
    
  
  cal <- wf %>%
    fit_resamples(vfold_cv(x_train %>%
                             mutate(y=as.factor(if_else(
                               y == 1,'pos','neg'))),
                           strata=y,v=all_cores),
                  metrics=metric_set(roc_auc,brier_class),
                  control=control_resamples(save_pred=TRUE)) %>%
    cal_estimate_logistic()
  
  y_hat <- wf %>%
    fit(data=x_train %>%
          mutate(y=as.factor(if_else(y == 1,'pos','neg')))) %>%
    augment(new_data=x_test %>%
              mutate(y=as.factor(if_else(y == 1,'pos','neg')))) %>%
    cal_apply(cal,pred_class=.pred_class) %>%
    select(pred_1=.pred_pos)
  
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
  
  res <- res %>%
    rows_update(as_tibble_row(perf) %>% 
                  mutate(lambda=lambda,
                         fn=lab),
                by='fn')
  
  print(res %>% select(-fn),n=Inf)
  
}

write_csv(res,file.path(path,'res','15_lasso_results.csv.gz'))

stopCluster(cl)
