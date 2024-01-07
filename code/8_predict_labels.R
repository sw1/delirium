library(tidymodels)
library(glmnet)
library(tidyverse)
library(doParallel)
library(vip)
library(rpart.plot)
library(ranger)

path <- 'D:\\Dropbox\\embeddings\\delirium'
mods <- c('rf','tree')
subsets <- c('icd','sub_chapter','major')

ds <- 'count_del'

m <- 'f_meas'

haobo_post <- read_rds(file.path(path,'data_tmp',sprintf('alldat_preprocessed_for_pred_%s.rds',ss)))

# change this to have rf and tree
tree_fit <- read_rds(file.path(path,'data_in',paste0('fit_tree_',ds,sprintf('_%s.rds',ss))))

# best_tree <- tree_fit$fit %>% select_by_pct_loss(metric=m,limit=5,tree_depth,desc(min_n))
best_tree <- tree_fit$fit %>% select_by_one_std_err(metric=m,tree_depth,desc(min_n))
wf <- tree_fit$wf %>% 
  finalize_workflow(best_tree) %>%
  last_fit(tree_fit$split) %>%
  extract_workflow()
# change this to load feature column
# add feature column output in other script
features <- colnames(wf$fit$fit$fit$model)[-1]

haobo_pred <- haobo_post %>%
  select(id,label,contains(features)) %>%
  mutate(across(everything(), ~replace_na(.x, 0))) %>%
  mutate(label=as.factor(label))

ids <- haobo_pred %>% select(id)
haobo_pred <- haobo_pred %>% select(-id)

# change this to if rf vs tree then predclass vs predprob
preds <- wf %>%
  predict(haobo_pred)

# change this to if rf vs tree then predclass vs predprob
labels <- ids %>% bind_cols(preds) %>% rename(label_tree=.pred_class)
print(table(labels$label_tree))

# change this to have rf and tree
# then fix script 9 to have new fn for loading
write_csv(labels,file.path(path,'data_in',paste0('labels_sptree_',ds,sprintf('_%s.csv.gz',ss))),
          col_names=TRUE)
