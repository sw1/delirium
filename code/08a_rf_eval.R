pacman::p_load(tidymodels,tidyverse,ranger,caret,rpart.plot,rpart)

# script to eval rf model from parameter sweep and fs

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

s1 <- 123

# get feature selection results
fs <- read_rds(file.path(path,'data_in','08_rf_fs.rds')) 

# select least complex model within 1 sd of best performing model based on rmse
params <- fs$perf %>%
  filter(rmse < min(rmse) + sd(rmse),
         n_feats < 200) %>%
  arrange(rmse) %>%
  slice_head(n=1)

feat_subset <- fs$features %>%
  arrange(desc(Importance)) %>%
  slice_head(n=params$n_feats[1]) %>%
  pull(Variable) %>%
  paste0('^',.,'$')

master <- read_rds(file.path(path,'data_in','07_rf_cv_params.rds'))$dat

train <- master %>% 
  filter(set == 'train') %>% 
  select(label,matches(feat_subset))

fit <- ranger(
  formula=label ~ ., 
  data=train, 
  num.trees=1500, 
  mtry=params$mtry[1],
  # max.depth=params$max_depth[1],
  min.node.size=floor(quantile(1:nrow(train),params$min_node_perc[1])),
  replace=TRUE, #params$replace[1],
  seed=s1,
  num.threads=1,
  importance='impurity',
  oob.error=TRUE,
  verbose=FALSE
)

y <- master %>% 
  filter(set == 'test') %>% 
  pull(label)

yhat <- predict(fit,master %>% 
                  filter(set == 'test') %>% 
                  select(label,matches(feat_subset)))$predictions

conf <- confusionMatrix(table(y,yhat),
                        mode='everything',
                        positive='1')

conf

master2 <- read_rds(file.path(path,'data_in','06_dat_rf_cv_fs.rds')) 

y <- master2 %>% 
  filter(set == 'heldout_expert') %>% 
  pull(label)

yhat <- predict(fit, master2 %>%
                  filter(set == 'heldout_expert') %>%
                  select(label,matches(feat_subset)))$predictions

conf <- confusionMatrix(table(y,yhat),
                        mode='everything',
                        positive='1')

conf


train <- master %>% 
  filter(set == 'train') %>% 
  select(label,matches(feat_subset))

set.seed(s1)
fit <- rpart(label ~., 
             data = train)

rpart.plot(fit)
