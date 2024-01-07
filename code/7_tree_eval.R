library(tidymodels)
library(glmnet)
library(tidyverse)
library(doParallel)
library(vip)
library(rpart.plot)

get_rules <- function(x){
  y <- matrix('',nrow=nrow(x),ncol=3)
  for (i in 1:nrow(x)){
    d <- as.numeric(x[i,1])
    y[i,3] <- d
    if (d < 0.5){
      y[i,1] <- paste0('v',100-i)
    }else{
      y[i,1] <- paste0('v',nrow(x)-i+1)
    }
    y[i,2] <- gsub('\\s+',' ',
                   paste0(
                     gsub('is','==',x[i,3:ncol(x)]),collapse=' '))
    y[i,2] <- paste0('if_else(',y[i,2],',1,0)')
    
  }
  return(y)
}

new_var <- function(df,v1,v2){
  df %>% mutate(!!rlang::parse_expr(v1) := !!rlang::parse_expr(v2))
}

update_df <- function(df,r){
  for (i in 1:nrow(r)){
    v1 <- r[i,1]
    v2 <- r[i,2]
    df <- new_var(df,v1,v2)
  }
  return(df)
}

get_icds <- function(x){
  z <- NULL
  for (i in 1:nrow(x)){
    r <- unlist(rules[i,])
    y <- r[startsWith(r,'icd_')]
    y <- gsub('icd_','',y)
    z <- c(z,y)
  }
  return(unique(z))
}



path <- 'D:\\Dropbox\\embeddings'

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

