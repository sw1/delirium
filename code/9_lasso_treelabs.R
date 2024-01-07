library(text2vec)
library(stopwords)
library(glmnet)
library(tidyverse)
library(doParallel)
library(caret)

z <- function(x) (x-mean(x,na.rm=TRUE))/sd(x,na.rm=TRUE)

path <- 'D:\\Dropbox\\embeddings\\delirium'

sets <- c('baseline','sub_chapter','major','icd')

for (s in sets){

  if (s == 'baseline'){
    dat <- read_csv(file.path(path,'to_python','tbl_to_python_updated.csv.gz'))
  }else{
    dat <- read_csv(file.path(path,'to_python',
                              sprintf('tbl_to_python_updated_count_del_%s.csv.gz',s))) 
  }
  
  dat <- dat %>%
    select(id,text=hpi_hc,label_h=label,label=label_icd,set)

  tbls <- list(
    train = dat %>% 
      filter(set %in% c('train','val')) %>% 
      select(id,text,label),
    test_icd_hpihc = dat %>% 
      filter(set == 'test_icd') %>% 
      select(id,text,label),
    test_haobo_hpihc_icd = dat %>% 
      filter(set == 'test_haobo') %>%
      select(id,text,label), 
    test_haobo_hpihc = dat %>% 
      filter(set == 'test_haobo') %>%
      select(id,text,label=label_h) 
  )
  
  if (s != 'baseline'){
    dat <- read_csv(file.path(path,'to_python',
                              sprintf('tbl_to_python_updated_treelabs_%s.csv.gz',s))) %>%
      select(id,text=hpi_hc,label_h=label)
    
    tbls[['test_treeheldout_haobo']] <- dat %>% 
      select(id,text,label=label_h)
  }
  
  cat(sprintf('Fitting lasso to labels from tree fit %s.\n',s))
  
  x_train <- readRDS(file.path(path,'data_in','doc_vectors_train.rds'))
  colnames(x_train) <- paste0('f_',1:ncol(x_train))
  
  ids <- intersect(as.character(tbls[['train']]$id),rownames(x_train))
  x_train <- x_train[ids,]
  
  y_train <- tbls[['train']]$label
  names(y_train) <- as.character(tbls[['train']]$id)
  y_train <- y_train[ids]
  
  w <- length(y_train)/(table(y_train) * length(unique(y_train)))
  w <- w[y_train+1]

  all_cores <- parallel::detectCores(logical = FALSE)
  cl <- makePSOCKcluster(all_cores)
  registerDoParallel(cl)

  set.seed(123)
  nfolds <- 10
  foldid <- sample(rep(seq(nfolds),length=nrow(x_train)))
  
  cat('Running lasso on training data.\n')
  fit <- cv.glmnet(x_train,as.factor(y_train),weights=w,
                   family='binomial',
                   type.measure='auc',
                   alpha=1,foldid=foldid,
                   standardize=TRUE,intercept=FALSE,
                   nfolds=nfolds,trace.it=1,keep=TRUE,
                   parallel=TRUE)
  
  stopCluster(cl)
  
  saveRDS(fit,file.path(path,'data_out',sprintf('lasso_meta_complete_%s.csv.rds',s)))
  
  fit <- readRDS(file.path(path,'data_out',sprintf('lasso_meta_complete_%s.csv.rds',s)))

  dv_test <- readRDS(file.path(path,'data_in','doc_vectors_test.rds'))
  
  tb_names <- names(tbls)[names(tbls) != 'train']
  cat(sprintf('\nLasso results for %s.\n\n',s))
  for (i in seq_along(tb_names)){
    
    x_test <- dv_test
    colnames(x_test) <- paste0('f_',1:ncol(x_test))
    
    ids <- intersect(as.character(tbls[[tb_names[i]]]$id),
                     rownames(x_test))
    x_test <- x_test[ids,]
    
    y_test <- tbls[[tb_names[i]]]$label
    names(y_test) <- as.character(tbls[[tb_names[i]]]$id)
    y_test <- y_test[ids]
    
    y_hat <- predict(fit,newx=x_test,s=fit$lambda.1se,type='class')
    
    cat(sprintf('\nTable %s.\n\n',tb_names[i]))
    print(confusionMatrix(table(y_test,y_hat)))
  }
}

stop('Analysis complete.')


#tree predict, haobo label, lasso predict
y2 <- tbls[[tb_names_full[2]]]$label
names(y2) <- as.character(tbls[[tb_names_full[2]]]$id)

y3 <- tbls[[tb_names_full[3]]]$label
names(y3) <- as.character(tbls[[tb_names_full[3]]]$id)

yhat <- y_hat
names(yhat) <- as.character(tbls[[tb_names_full[2]]]$id)

ids <- intersect(names(y2),names(y3))
y2 <- y2[ids]
y3 <- y3[ids]
yhat <- yhat[ids]

test <- cbind(y2,y3,yhat)

#correct prediction of delirium based on tree predicted delirium 
#but haobo says no delirum
mismatch_tbl <- tibble(id=rownames(test),as_tibble(test)) %>%
  rename(tree=y2,haobo=y3,lasso=yhat) %>%
  filter(tree == lasso,
         tree != haobo) %>%
  left_join(tbls[[tb_names_full[2]]] %>% mutate(id=as.character(id)),by='id') %>%
  arrange(desc(haobo))

dat <- read_rds(file.path(path,paste0('fit_',ds,'.rds')))

dat$data %>% filter(id == 15594       ) %>% 
  select(starts_with('count'),starts_with('icd_')) %>%
  # select(-icd_codes_del) %>%
  gather(feature,count) %>%
  filter(count > 0) %>%
  print(n=100)

mismatch_tbl %>% filter(id == 15594       ) %>% 
  select(text) %>% unlist()

#2560: alc --> both
#2582: alc --> both
#2633: schizo, crazy
#10780: met enceph, very soft call

conc <- read_csv(file.path(path,'concordance_labels.csv')) %>%
  rename(case=internalcaseid_deid_rdr)

haobo <- readRDS(file.path(path,'tbl_final_wperiods.rds')) 

tbl_full <- read_csv(file.path(path,'Discharge Summaries ID+ ICD.csv')) %>%
  filter(note_type == 'DISCHARGE SUMMARY') %>%
  select(id=ïnote_id,case=internalcaseid_deid_rdr,
         mrn=mrn_n,note=note_txt,icd=delirium_ICD) %>%
  left_join(conc,by='case')

tbl_full %>% filter(mrn == '2655321') %>% select(case)
tbl_full %>% filter(mrn == '2655321') %>% select(note) %>% unlist()



extract_workflow(dat$final_fit) %>%
  extract_fit_engine() %>%
  rpart.plot(type=4, 
             extra = 101, 
             branch.lty=3,
             nn=F,
             roundint=F, cex = 0.6, under=F,
             box.palette = "auto",clip.right.labs = F,fallen.leaves = T)

extract_workflow(dat$final_fit)  %>%
  extract_fit_engine() %>%
  rpart.rules()
