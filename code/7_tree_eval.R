library(tidymodels)
library(glmnet)
library(tidyverse)
library(doParallel)
library(vip)
library(rpart.plot)
library(ranger)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

subsets <- c('sub_chapter','major','icd')
mods <- c('rf','tree')

ss <- subsets[3]
mod <- mods[1]

mets <- metric_set(accuracy, sens, yardstick::spec, f_meas, roc_auc)

tree_fit <- read_rds(file.path(path,'data_in',sprintf('fit_%s_count_del_%s.rds',mod,ss)))

haobo_pre <- tree_fit$data
m <- 'f_meas'

if (mod == 'rf'){
  best_tree <- tree_fit$fit %>% select_by_pct_loss(metric=m,limit=5,desc(min_n),trees,desc(mtry))
}
if (mod == 'tree'){
  best_tree <- tree_fit$fit %>% select_by_pct_loss(metric=m,limit=5,desc(min_n),tree_depth)
}

tree_fit$fit %>% show_best(m)
best_tree

final_fit <- tree_fit$wf %>% 
  finalize_workflow(best_tree) %>%
  last_fit(tree_fit$split) 

preds <- final_fit %>% collect_predictions()

tree_fit$fit %>% collect_metrics()
mets(preds, truth = label,estimate = .pred_class, .pred_0)
mets(preds %>% 
       mutate(.pred_class=as.factor(if_else(.pred_1 >= 0.65,1,0))), 
     truth = label,estimate = .pred_class, .pred_0)

if (mod == 'tree'){
  tree_fit$fit %>%
    collect_metrics() %>%
    mutate(tree_depth = factor(tree_depth),
           min_n = factor(min_n)) %>%
    ggplot(aes(cost_complexity, mean, color = tree_depth)) +
    geom_line(linewidth = .5, alpha = 0.6) +
    geom_point(size = 2) +
    facet_grid(.metric ~ min_n) +
    scale_x_log10(labels = scales::label_number()) +
    scale_color_viridis_d(option = "plasma", begin = .9, end = 0)
}
if (mod == 'rf'){
  tree_fit$fit %>%
    collect_metrics() %>%
    mutate(trees = factor(trees),
           min_n = factor(min_n),
           mtry = mtry) %>%
    ggplot(aes(mtry, mean, color = min_n)) +
    geom_line(linewidth = .5, alpha = 0.6) +
    geom_point(size = 2) +
    facet_grid(.metric ~ trees) +
    # scale_x_log10(labels = scales::label_number()) +
    scale_color_viridis_d(option = "plasma", begin = .9, end = 0)
}


final_fit %>% collect_metrics()
final_fit %>%
  collect_predictions() %>% 
  roc_curve(label, .pred_0) %>% 
  autoplot()

if (mod == 'tree'){
  final_tree <- extract_workflow(final_fit)
  final_tree %>%
    extract_fit_engine() %>%
    rpart.plot(extra=1,cex=0.5,type=3,clip.right.labs=FALSE)  
  
  final_tree %>% 
    extract_fit_parsnip() %>% 
    vip()
}

