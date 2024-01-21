library(text2vec)
library(stopwords)
library(glmnet)
library(tidyverse)
library(doParallel)
library(caret)
library(glue)

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

fns <- list.files(file.path(path,'to_python'))
fns <- fns[str_detect(fns,'chunked')] 
fns <- fns[!str_detect(fns,'treeheldout')]
fns <- fns[!str_detect(fns,'zeroed')]


#Fitting lasso to rfst sub 855 50

dat_expert <- read_csv(
  file.path(path,'to_python',
            'tbl_to_python_updated_chunked_treeheldout.csv.gz')) %>%
  select(id,text=hpi_hc,label_h=label,label=label_icd,set)

res <- tibble()
for (j in seq_along(fns)){

  mods <- str_split(fns[j],'_')[[1]][6:10]
  
  tmp <- tibble(mod=if_else(all(is.na(mods)),'baseline',mods[1]),
                icd_type=str_match(mods[2],'s(.*)')[1,2],
                threshold=str_match(mods[3],'t(\\d+)')[1,2],
                weight=str_match(mods[4],'w(.*)')[1,2],
                n_features=str_match(mods[5],'f(.*)')[1,2])

  dat <- read_csv(file.path(path,'to_python',fns[j])) %>%
    select(id,text=hpi_hc,label_h=label,label=label_icd,set)

  tbls <- list(
    train = dat %>% 
      filter(set %in% c('train','val')) %>% 
      select(id,text,label),
    test_pseudo = dat %>% 
      filter(set == 'test_icd') %>% 
      select(id,text,label),
    test_pseudo_exp = dat %>% 
      filter(set == 'test_haobo') %>%
      select(id,text,label), 
    test_exp = dat %>% 
      filter(set == 'test_haobo') %>%
      select(id,text,label=label_h), 
    ho = dat_expert %>%
      filter(set == 'test_haobo') %>%
      select(id,text,label)
  )
  
  if (any(sapply(tbls,nrow) == 0)) next
  
  cat(glue('({j}) fitting lasso to {tmp$mod} {tmp$icd_type} {tmp$threshold} \\
           {tmp$weight} {tmp$n_features}\n
           ',.na=''))
  
  x_train <- readRDS(file.path(path,'data_in','doc_vectors_train.rds'))
  colnames(x_train) <- paste0('f_',1:ncol(x_train))
  
  ids_train <- unique(intersect(tbls[['train']]$id,rownames(x_train)))
  x_train <- x_train[rownames(x_train) %in% ids_train,]
  
  y_train <- tibble(id=rownames(x_train)) %>% 
    left_join(tbls[['train']] %>% 
                select(id,label) %>% 
                mutate(id=as.character(id)) %>% 
                distinct(), 
              by='id') %>%
    select(label) %>% 
    unlist()
  
  if (length(table(y_train)) == 1) next
  
  w <- length(y_train)/(table(y_train) * length(unique(y_train)))
  w <- w[y_train+1]

  set.seed(123)
  nfolds <- 10
  foldid <- sample(rep(seq(nfolds),length=nrow(x_train)))
  
  # cat('Running lasso on training data.\n')
  fit <- cv.glmnet(x_train,as.factor(unname(y_train)),weights=w,
                   family='binomial',
                   type.measure='auc',
                   alpha=1,foldid=foldid,
                   standardize=TRUE,intercept=FALSE,
                   nfolds=nfolds,trace.it=0,keep=TRUE,
                   parallel=TRUE)
  
  
  dv_test <- readRDS(file.path(path,'data_in','doc_vectors_test.rds'))
  
  tb_names <- names(tbls)[names(tbls) != 'train']
  
  perf <- NULL
  for (i in seq_along(tb_names)){
    
    x_test <- dv_test
    colnames(x_test) <- paste0('f_',1:ncol(x_test))
    
    ids_test <- unique(intersect(tbls[[tb_names[i]]]$id,rownames(x_test)))
    x_test <- x_test[rownames(x_test) %in% ids_test,]
    
    y_test <- tibble(id=rownames(x_test)) %>% 
      left_join(tbls[[tb_names[i]]] %>% 
                  select(id,label) %>% 
                  mutate(id=as.character(id)) %>% 
                  distinct(), 
                by='id') %>%
      select(label) %>% 
      unlist()
    
    y_hat <- predict(fit,newx=x_test,s=fit$lambda.1se,type='class')
    
    conf <- confusionMatrix(table(y_test,y_hat),
                          reference='y_test',
                          mode='everything',positive='1')
    perf_tmp <- c(conf$byClass[11],
                  conf$byClass[5],
                  conf$byClass[6],
                  conf$byClass[7],
                  table(y_hat)[2]/length(y_hat))
    names(perf_tmp) <- paste0(tb_names[i],'_',
                              c('acc','prec','recall','f1','prop1'))
    
    print(perf_tmp)
    perf <- c(perf,perf_tmp)
  }
  tmp <- tmp %>% bind_cols(as_tibble_row(perf))
  res <- res %>% bind_rows(tmp)
}

write_csv(res,file.path(path,'res','lasso_results.csv.gz'))

stopCluster(cl)
