pacman::p_load(tidymodels,tidyverse,doParallel,
               vip,icd.data,ranger,glue,caret)

# script to perform initial random forest for self training on expert
# labeled notes only for parameter cross validation after upsamp

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
  all_cores <- 4
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
  all_cores <- 8
}
source(file.path(path,'code','fxns.R'))

outfile <- file.path(path,'..\\featsel_out.txt')
file.remove(outfile)

#all_cores <- parallel::detectCores(logical=TRUE)
cl <- makePSOCKcluster(all_cores,outfile=outfile)
registerDoParallel(cl)

s1 <- 2 # feature fit
s2 <- 23 # cv split seed
n_folds <- 5 # cv folds for feature selection

p_feats <- c(0.99,0.975,0.95,0.9,0.85) # percentiles of feat imp to test

dat <- read_rds(file.path(path,'data_in','07_rf_cv_params.rds'))

master <- dat$dat

# filtering essentially ties
# params <- dat$params %>%
#   arrange(rmse,desc(min_node),mtry,desc(replace),reg_factor) %>%
#   mutate(rmse_round=as.character(round(rmse,3))) %>%
#   group_by(rmse_round) %>%
#   filter(row_number() == 1) %>%
#   ungroup() %>%
#   select(-rmse_round)

params <- dat$params %>%
  arrange(rmse) %>%
  slice_head(n=1)

rm(dat)
gc()

# build final model with best parameters with importance measure for
# feature selection. Using top model since features are approx the same
# irrespective of model
fit <- ranger(
  formula=label ~ ., 
  data=master %>% 
    filter(set == 'train') %>% 
    select(-id,-set), 
  num.trees=1500, 
  mtry=params$mtry[1],
  # max.depth=params$max_depth[1],
  min.node.size=params$min_node[1],
  replace=TRUE,
  regularization.factor=params$reg_factor[1],
  regularization.usedepth=TRUE,
  seed=s1,
  num.threads=1,
  oob.error=FALSE,
  importance='impurity',
  write.forest=FALSE,
  verbose=FALSE
)

# extract features breaks based on percentiles
feature_imp <- importance(fit) 
feature_imp <- tibble(Variable=names(feature_imp),
                      Importance=feature_imp) %>%
  arrange(desc(Importance))

n_feats <- sapply(quantile(feature_imp$Importance,p_feats),function(p){
  feature_imp %>% 
    filter(Importance >= p) %>% 
    nrow()
})

features <- feature_imp %>% 
  pull(Variable)

combs <- as_tibble(expand.grid(
  min_node_perc = c(0.001,0.005,0.01,0.015,0.02),
  # max_depth = c(3,5,7,10),
  # replace = TRUE, # consider F given 1-hot-coding and upsamp
  n_feats = n_feats,
  mtry = seq(3,floor(max(sqrt(n_feats))),length.out=4)
)) %>%
  filter(mtry <= sqrt(n_feats))

perf <- foreach(i=1:nrow(combs),.combine='rbind',
                .errorhandling='remove',
                .packages=c('tidyverse','caret',
                            'ranger','glue')) %dopar% {
    
  # subset features
  feat_subset <- paste0('^',features[1:combs$n_feats[i]],'$')
  
  train <- master %>% 
    filter(set == 'train') %>% 
    select(label,matches(feat_subset))
  
  fit <- ranger(
    formula=label ~ ., 
    data=train, 
    num.trees=1500, 
    mtry=combs$mtry[i],
    # max.depth=combs$max_depth[i],
    min.node.size=get_node_size(combs$min_node_perc[i],train),
    replace=TRUE,
    seed=s2,
    num.threads=1,
    oob.error=TRUE,
    verbose=FALSE
  )
  
  sqrt_err <- sqrt(fit$prediction.error)
  
  y <- master %>% 
    filter(set == 'test') %>% 
    pull(label)
  
  yhat <- predict(fit,master %>% 
                    filter(set == 'test') %>% 
                    select(label,matches(feat_subset)))$predictions
  
  conf <- confusionMatrix(table(y,yhat),
                          mode='everything',
                          positive='1')
  perf_tmp <- c(conf$byClass[11],
                conf$byClass[5],
                conf$byClass[6],
                conf$byClass[7],
                table(yhat)[2]/length(y))
  names(perf_tmp) <- c('bacc','prec','rec','f1','prop1')
  
  perf <- enframe(perf_tmp) %>%
    pivot_wider(names_from=name,values_from=value) %>%
    mutate(rmse=sqrt_err) %>%
    bind_cols(combs[i,])
  
  perf_round <- perf %>% 
    mutate(across(where(is.numeric),~round(.x,4))) 
  cat(glue('\n\n(i={i}): mtry={perf_round$mtry[1]}, ',
           'min_node_perc={perf_round$min_node_perc[1]}, ',
           'n_feats={perf_round$n_feats[1]}, ',
           'rmse: {perf_round$rmse[1]}, ',
           'test acc: {perf_round$bacc[1]}\n'))
  
  return(perf)
  
}

cat('Saving output.')

write_rds(list(features=feature_imp,
               perf=perf), 
          file.path(path,'data_in','08_rf_fs.rds'))

stopCluster(cl)



