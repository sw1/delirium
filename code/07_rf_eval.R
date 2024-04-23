library(tidymodels)
library(glmnet)
library(tidyverse)
library(doParallel)
library(vip)
library(rpart.plot)
library(ranger)

# script to eval rf model from parameter sweep and cv

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

mets <- metric_set(accuracy, sens, yardstick::spec, f_meas, roc_auc)
m <- 'f_meas'

tree_fit <- read_rds(file.path(path,'data_in','06_fit_st_rf.rds'))

best_tree <- tree_fit$fit %>% 
  select_by_pct_loss(metric=m,limit=5,desc(min_n),desc(mtry))

tree_fit$fit %>% show_best(m,n=20)
best_tree

final_fit <- tree_fit$wf %>% 
  finalize_workflow(best_tree) %>%
  last_fit(tree_fit$split) 

preds <- final_fit %>% 
  collect_predictions()

tree_fit$fit %>% collect_metrics()
mets(preds, truth = label,estimate = .pred_class, .pred_0)
mets(preds %>% 
       mutate(.pred_class=as.factor(if_else(.pred_1 >= 0.65,1,0))), 
     truth = label,estimate = .pred_class, .pred_0)

tree_fit$fit %>%
  collect_metrics() %>%
  mutate(lambda = factor(lambda),
         min_n = factor(min_n),
         depth = factor(as.integer(depth)),
         mtry = mtry) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(linewidth = .5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_grid(.metric ~ lambda + depth) +
  scale_color_viridis_d(option='plasma', begin = .9, end = 0)

final_fit %>% collect_metrics()
final_fit %>%
  collect_predictions() %>% 
  roc_curve(label, .pred_0) %>% 
  autoplot()

tree_fit$features %>%
  print(n=125)
