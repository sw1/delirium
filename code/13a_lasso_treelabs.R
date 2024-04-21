library(text2vec)
library(stopwords)
library(glmnet)
library(tidyverse)
library(doParallel)
library(caret)
library(glue)

# script to perform lasso on doc embeddings

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

all_cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

# params
s1 <- 123 # seed for lasso
 
fns <- list.files(file.path(path,'to_python'),
                  pattern='tbl_to_python_expertupdate_chunked')
  
res <- tibble()
fits <- list()
for (j in seq_along(fns)){
  for (train_on_expert in c(TRUE,FALSE)){ # expert labeled samples in training
  
    # extract params
    mod <- if_else(!str_detect(fns[j],'rfst'),'bl','rfst')
    seed <- case_when(mod == 'bl' ~ 'bl',
                      str_detect(fns[j],'majvote') ~ 'majvote',
                      str_detect(fns[j],'seed') ~ str_match(fns[j],
                                                            'seed(\\d+)')[1,2],
                      TRUE ~ NA)
    th <- if_else(mod == 'bl','bl',str_match(fns[j],'th(\\d+)')[1,2])
    f <- if_else(mod == 'bl','bl',str_match(fns[j],'nfeat(\\d+)')[1,2])
  
    fn_suffix <- glue('{mod}_seed{seed}_th{th}_',
                      'f{f}_toe{as.integer(train_on_expert)}')
    
    dat <- read_csv(file.path(path,'to_python',fns[j]))
  
    # remove sets, note that training will include both training
    # samples and the remaining expert labeled samples (with their
    # true labels) that werent used for testing. Baseline model
    # uses ICD labels except for expert labeled samples which use
    # true labels.
    if (mod == 'bl'){
      if (train_on_expert){
        train <- dat %>% 
          mutate(label_icd=if_else(set == 'train_expert',label,label_icd)) 
      }
      train <- dat %>% 
        filter(set == 'train' | set == 'train_expert') %>% 
        select(id,text=hpi_hc,label=label_icd)
      val <- dat %>%
        filter(set == 'val') %>%
        select(id,text=hpi_hc,label=label_icd)
    }else{
      if (train_on_expert){
        train <- dat %>% 
          mutate(label_pseudo=if_else(set == 'train_expert',label,label_pseudo)) 
      }
      train <- dat %>% 
        filter(set == 'train' | set == 'train_expert') %>% 
        select(id,text=hpi_hc,label=label_pseudo)
      val <- dat %>%
        filter(set == 'val') %>%
        select(id,text=hpi_hc,label=label_pseudo)
    }
    # remove unlabeled samples
    train <- train %>%
      filter(label %in% c(0,1))
    val <- val %>%
      filter(label %in% c(0,1))
    # pull test sets
    test_icd <- dat %>% 
      filter(set == 'test_icd') %>% 
      select(id,text=hpi_hc,label=label_icd)
    test_expert <- dat %>% 
      filter(set == 'test_expert') %>%
      select(id,text=hpi_hc,label)
    test_heldout <- dat %>% 
      filter(set == 'heldout_expert') %>%
      select(id,text=hpi_hc,label)
    
    cat(glue('\n({j}) fitting lasso: mod={mod}, seed={seed}, threshold={th}, ',
             'features={f}, train_on_expert={as.integer(train_on_expert)}\n\n',
             .na=''))
    
    # load training doc vectors
    x_train <- readRDS(file.path(path,'data_in','doc_vectors_train.rds'))
    if (train_on_expert){
      # load testing doc vectors (which includes all expert labeled samples)
      # and then add the expert labeled samples to the training set
      x_train <- rbind(x_train,
                       readRDS(file.path(path,'data_in','doc_vectors_test.rds')))
    }
    colnames(x_train) <- paste0('f_',1:ncol(x_train))
    # ensure overlapping samples
    ids_train <- train %>% pull(id) %>% unique()
    x_train <- x_train[rownames(x_train) %in% ids_train,]
    
    y_train <- tibble(id=as.numeric(rownames(x_train))) %>% 
      left_join(train %>% 
                  select(id,label) %>%
                  distinct(), 
                by='id') %>%
      select(label) %>% 
      unlist()
  
    # calculate weights for class imbalance
    w <- length(y_train)/(table(y_train) * length(unique(y_train)))
    w <- w[y_train+1]
  
    set.seed(s1)
    nfolds <- 10
    foldid <- sample(rep(seq(nfolds),length=nrow(x_train)))
    
    # fit lasso
    fit <- cv.glmnet(x_train,as.factor(unname(y_train)),weights=w,
                     family='binomial',
                     type.measure='auc',
                     alpha=1,foldid=foldid,
                     standardize=TRUE,intercept=FALSE,
                     nfolds=nfolds,trace.it=0,keep=TRUE,
                     parallel=TRUE)
    
    fits <- c(fits,list(fit))
    names(fits)[length(fits)] <- fn_suffix
    
    # load testing doc vectors
    x_test <- readRDS(file.path(path,'data_in','doc_vectors_test.rds'))
    colnames(x_test) <- paste0('f_',1:ncol(x_test))
    
    # build test tables and labels
    ids_test_icd <- dat %>% 
      filter(set == 'test_icd') %>% 
      pull(id) %>% 
      unique()
    x_test_icd <- x_test[rownames(x_test) %in% ids_test_icd,]
    y_test_icd <- tibble(id=as.numeric(rownames(x_test_icd))) %>%
      left_join(dat %>% select(id,label_icd) %>% distinct(),by='id') %>% 
      pull(label_icd)
    
    ids_test_expert <- dat %>% 
      filter(set == 'test_expert') %>% 
      pull(id) %>% 
      unique()
    x_test_expert <- x_test[rownames(x_test) %in% ids_test_expert,]
    y_test_expert <- tibble(id=as.numeric(rownames(x_test_expert))) %>%
      left_join(dat %>% select(id,label) %>% distinct(),by='id') %>% 
      pull(label)
    
    ids_test_heldout <- dat %>% 
      filter(set == 'heldout_expert') %>% 
      pull(id) %>% 
      unique()
    x_test_heldout <- x_test[rownames(x_test) %in% ids_test_heldout,]
    y_test_heldout <- tibble(id=as.numeric(rownames(x_test_heldout))) %>%
      left_join(dat %>% select(id,label) %>% distinct(),by='id') %>% 
      pull(label)
    
    # create list of tables to loop results
    x_tbls <- list(test_icd=x_test_icd,test_expert=x_test_expert,
                   test_heldout=x_test_heldout)
    y_tbls <- list(test_icd=y_test_icd,test_expert=y_test_expert,
                   test_heldout=y_test_heldout)
    
    # predict and calculate performance
    perf <- NULL
    for (i in seq_along(x_tbls)){
      y_hat <- predict(fit,newx=x_tbls[[i]],s=fit$lambda.1se,type='response')
      
      # get majority vote since chunked
      mv <- majority_vote(y_hat,y_tbls[[i]])
      
      # get performance
      conf <- confusionMatrix(table(mv$lab,mv$pred),
                            mode='everything',
                            positive='1')
      perf_tmp <- c(conf$byClass[11],
                    conf$byClass[5],
                    conf$byClass[6],
                    conf$byClass[7],
                    table(mv$pred)[2]/length(mv$pred))
      names(perf_tmp) <- c('bacc','prec','rec','f1','prop1')
      
      perf <- rbind(perf,perf_tmp)
    }
    
    # bind results
    perf <- tibble(set=str_match(names(y_tbls),'test_(.*)')[,2],
           mod=mod,seed=seed,th=th,f=f,toe=as.integer(train_on_expert)) %>%
      bind_cols(as_tibble(perf))
    res <- res %>% bind_rows(perf)
    print(res,n=Inf)
  }
}

write_csv(res,file.path(path,'res','lasso_results.csv.gz'))
write_rds(fits,file.path(path,'res','lasso_fits.rds'))

stopCluster(cl)
