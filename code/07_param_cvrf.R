pacman::p_load(tidymodels,tidyverse,doParallel,vip,icd.data,ranger,glue)

# script to perform initial parameter sweep for feature selection via rf

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
  all_cores <- 4
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
  all_cores <- 8
}
source(file.path(path,'code','fxns.R'))

outfile <- file.path(path,'scratch','paramcv_out.txt')
file.remove(outfile)

#all_cores <- parallel::detectCores(logical=TRUE)
cl <- makePSOCKcluster(all_cores,outfile=outfile)
registerDoParallel(cl)

s0 <- 235
s1 <- 7829 # seed for upsamp
s2 <- 123 # seed for rf sweep

master <- read_rds(file.path(path,'data_out','06_dat_rf_cv_fs.rds'))

# # take out testing and heldout to upsample training data
# heldout <- master %>%
#   filter(set == 'heldout_expert')
# 
# test <- master %>%
#   filter(set == 'test_expert')
# 
# master <- master %>%
#   anti_join(heldout,by='id') %>%
#   anti_join(test,by='id')
# 
# # upsample smaller class for balanced training
# set.seed(s1)
# master <- upsamp(master)
# 
# # rebind heldout set
# master <- master %>%
#   bind_rows(test) %>%
#   bind_rows(heldout)

d_train <- master %>% 
  filter(set == 'expert') %>% 
  select(-id,-set)

params <- as_tibble(expand.grid(
  mtry = c(5,10,15), # plan to ultimately feat select so sqrt(250) is max
  min_node = c(1),
  reg_factor = c(1e-10,1e-7)
))

gc()

out <- foreach(i=1:nrow(params),.combine='rbind',
               .packages=c('tidyverse','ranger','glue')) %dopar% {
                 
  fit <- ranger(
    formula=label ~ ., 
    data=d_train, 
    num.trees=1500, 
    mtry=params$mtry[i],
    # max.depth=params$max_depth[i],
    min.node.size=params$min_node[i],
    # replace=params$replace[i],
    regularization.factor=params$reg_factor[i],
    regularization.usedepth=TRUE,
    seed=s2,
    num.threads=1,
    write.forest=FALSE,
    verbose=FALSE
  )
  
  sqrt_err <- sqrt(fit$prediction.error)
  
  cat(glue('\n(i={i}) RF: mtry={params$mtry[i]}, ',
           'min_node={params$min_node[i]}, ',
           'reg_factor={params$reg_factor[i]}\n',
           '\trmse: {round(sqrt_err,4)}\n\n'))
  
  return(params[i,] %>% mutate(rmse=sqrt_err))
  
}

cat('Saving output.')

write_rds(list(params=out,
               dat=master), 
          file.path(path,'data_out','07_rf_cv_params.rds'))

stopCluster(cl)

