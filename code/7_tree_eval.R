library(tidymodels)
library(glmnet)
library(tidyverse)
library(doParallel)
library(vip)
library(rpart.plot)

path <- 'D:\\Dropbox\\embeddings'
source(file.path(path,'code','fxns.R'))

labs <- c('sub_chapter','major','icd')

lab <- labs[3]

mets <- metric_set(accuracy, sens, yardstick::spec, f_meas, roc_auc)

tree_fit <- read_rds(file.path(path,'data_in',sprintf('fit_tree_count_del_%s.rds',lab)))

haobo_pre <- tree_fit$data
m <- 'f_meas'

best_tree <- tree_fit$fit %>% select_by_one_std_err(metric=m,tree_depth,desc(min_n))
final_fit <- tree_fit$wf %>% 
  finalize_workflow(best_tree) %>%
  last_fit(tree_fit$split) 

preds <- final_fit %>% collect_predictions()

tree_fit$fit %>% collect_metrics()
mets(preds, truth = label,estimate = .pred_class, .pred_0)

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

tree_fit$fit %>% show_best(m)
tree_fit$fit %>% select_by_pct_loss(metric=m,limit=5,desc(n))

final_fit %>% collect_metrics()
final_fit %>%
  collect_predictions() %>% 
  roc_curve(label, .pred_0) %>% 
  autoplot()

final_tree <- extract_workflow(final_fit)
final_tree %>%
  extract_fit_engine() %>%
  rpart.plot(extra=1,cex=0.5,type=3,clip.right.labs=FALSE)

final_tree %>% 
  extract_fit_parsnip() %>% 
  vip()

