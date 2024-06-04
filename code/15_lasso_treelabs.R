pacman::p_load(text2vec,stopwords,glmnet,tidyverse,doParallel,caret,glue)

# script to perform lasso on doc embeddings

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
  all_cores <- 4
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
  all_cores <- 8
}
source(file.path(path,'code','fxns.R'))

#all_cores <- parallel::detectCores(logical=FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

# params
s1 <- 123 # seed for lasso
 
params <- tibble(fn=list.files(file.path(path,'to_python'),
                  pattern='tbl_to_python_expertupdate_')) %>%
  mutate(mod=if_else(!str_detect(fn,'rfst'),'bl','rfst'),
         mod=case_when(str_detect(fn,'rfst') ~ 'rfst',
                       str_detect(fn,'onlyexpert') ~ 'bl_onlyexpert',
                       str_detect(fn,'fullexpert') ~ 'bl_fullexpert',
                       TRUE ~ 'bl'),
         seed=case_when(str_detect(mod,'bl') ~ NA,
                        str_detect(fn,'majvote') ~ 'majvote',
                        str_detect(fn,'seed') ~ 
                          str_match(fn,'seed(\\d+)')[,2],
                        TRUE ~ NA),
         th=if_else(str_detect(mod,'bl'),NA,str_match(fn,'th(\\d+)')[,2]),
         ns=if_else(str_detect(mod,'bl'),NA,str_match(fn,'ns(\\d+)')[,2])) %>%
  crossing(toe=c(T,F)) %>%
  filter(!(str_detect(mod,'bl_') & toe == T)) %>%
  arrange(mod) %>%
  mutate(fn_suffix=case_when(
    str_detect(mod,'bl_') ~ glue('{mod}'),
    str_detect(mod,'bl') ~ glue('{mod}_toe{as.integer(toe)}'),
    str_detect(mod,'rfst') ~ glue('{mod}_seed{seed}_th{th}_',
                                  'ns{ns}_toe{as.integer(toe)}'),
    TRUE ~ NA))

res <- tibble()
fits <- list()
for (j in 1:nrow(params)){
  
  dat <- read_csv(file.path(path,'to_python',params$fn[j]))

  # remove baseline sets
  if (str_detect(params$mod[j],'bl')){
    # remove baseline sets using expert labels, either expert/concordance 1/0s
    # or expert with unlabeled set to 0
    if (str_detect(params$fn[j],'fullexpert|onlyexpert')){
      train <- dat %>% 
        filter(set == 'train') %>% 
        select(id,text=hpi_hc,label=label)
      val <- dat %>%
        filter(set == 'val') %>%
        select(id,text=hpi_hc,label=label)
    }else{
      # for icd baseline, use expert labels in training
      if (params$toe[j]){
        train <- dat %>% 
          filter(set == 'train' | set == 'train_expert') %>% 
          select(id,text=hpi_hc,label=label_icd)
        val <- dat %>%
          filter(set == 'val') %>%
          select(id,text=hpi_hc,label=label_icd)
      }else{
        # dont use expert labels
        train <- dat %>% 
          filter(set == 'train') %>% 
          select(id,text=hpi_hc,label=label_icd)
        val <- dat %>%
          filter(set == 'val') %>%
          select(id,text=hpi_hc,label=label_icd)
      }
    }
  }else{
    # sets for rfst pseudolabels, starting with those using remaining expert
    # labels in training
    if (params$toe[j]){
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
  # for expert baseline sets, only heldout matters
  test_icd <- dat %>% 
    filter(set == 'test_icd') %>% 
    select(id,text=hpi_hc,label=label_icd)
  test_expert <- dat %>% 
    filter(set == 'test_expert') %>%
    select(id,text=hpi_hc,label)
  test_heldout <- dat %>% 
    filter(set == 'heldout_expert') %>%
    select(id,text=hpi_hc,label)
  
  cat(glue('\n({j}) fitting lasso: mod={params$mod[j]}, ',
           'seed={params$seed[j]}, threshold={params$th[j]}, ',
           'ns_adapt={params$ns[j]}, ',
           'train_on_expert={as.integer(params$toe[j])}\n\n',
           .na=NA))
  
  # load training doc vectors
  if (str_detect(params$mod[j],'fullexpert|onlyexpert')){
    x_train <- read_rds(
      file.path(path,'data_in',
                glue("doc_vectors_train",
                     "{str_replace(params$mod[j],'bl','')}.rds")))
  }else{
    x_train <- read_rds(file.path(path,'data_in','doc_vectors_train.rds')) 
  }
  
  if (params$toe[j]){
    # load testing doc vectors (which includes all expert labeled samples)
    # and then add the expert labeled samples to the training set 
    # then filter for intersection
    x_train <- rbind(x_train,
                     read_rds(
                       file.path(path,'data_in','doc_vectors_test.rds')))
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
  names(fits)[length(fits)] <- params$fn_suffix[j]
  
  # load testing doc vectors
  if (str_detect(params$mod[j],'fullexpert|onlyexpert')){
    x_test <- read_rds(
      file.path(path,'data_in',
                glue("doc_vectors_test",
                     "{str_replace(params$mod[j],'bl','')}.rds")))
  }else{
    x_test <- read_rds(file.path(path,'data_in','doc_vectors_test.rds')) 
  }
  colnames(x_test) <- paste0('f_',1:ncol(x_test))
  
  
  # build heldout table which applies to all models
  ids_test_heldout <- dat %>% 
    filter(set == 'heldout_expert') %>% 
    pull(id) %>% 
    unique()
  x_test_heldout <- x_test[rownames(x_test) %in% ids_test_heldout,]
  y_test_heldout <- tibble(id=as.numeric(rownames(x_test_heldout))) %>%
    left_join(dat %>% select(id,label) %>% distinct(),by='id') %>% 
    pull(label)
  
  
  # build test tables for rfst and baseline
  if (!str_detect(params$mod[j],'fullexpert|onlyexpert')){
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
    
    x_tbls <- list(test_icd=x_test_icd,test_expert=x_test_expert,
                   test_heldout=x_test_heldout)
    y_tbls <- list(test_icd=y_test_icd,test_expert=y_test_expert,
                   test_heldout=y_test_heldout)
  }else{
    x_tbls <- list(test_heldout=x_test_heldout)
    y_tbls <- list(test_heldout=y_test_heldout)
  }
  
  # predict and calculate performance
  perf <- NULL
  for (i in seq_along(x_tbls)){
    y_hat <- predict(fit,newx=x_tbls[[i]],s=fit$lambda.1se,type='response')
    
    pre_perf <- tibble(id=rownames(y_hat),
                       pred=if_else(y_hat[,1] >= 0.5,1,0),
                       lab=y_tbls[[i]])
    
    pre_conf <- confusionMatrix(table(pre_perf$lab,pre_perf$pred),
                                mode='everything',
                                positive='1')
    
    pre_perf_tmp <- c(pre_conf$byClass[11],
                      pre_conf$byClass[5],
                      pre_conf$byClass[6],
                      pre_conf$byClass[7],
                      table(pre_perf$pred)[2]/length(pre_perf$pred))
    names(pre_perf_tmp) <- c('bacc','prec','rec','f1','prop1')
    
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
    names(perf_tmp) <- c('mv_bacc','mv_prec','mv_rec','mv_f1','mv_prop1')
    
    perf <- rbind(perf,c(pre_perf_tmp,perf_tmp))
  }
  
  # bind results
  perf <- tibble(set=str_match(names(y_tbls),'test_(.*)')[,2],
         mod=params$mod[j],seed=params$seed[j],th=params$th[j],
         ns=params$ns[j],toe=as.integer(params$toe[j])) %>%
    bind_cols(as_tibble(perf))
  
  res <- res %>% 
    bind_rows(perf) %>%
    arrange(set,desc(bacc))
  
  print(res %>% select(set:toe,bacc,f1,prop1,mv_bacc,mv_f1),
        n=Inf)
  
}

write_csv(res,file.path(path,'res','lasso_results.csv.gz'))
write_rds(fits,file.path(path,'res','lasso_fits.rds'))

stopCluster(cl)
