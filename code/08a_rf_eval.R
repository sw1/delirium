pacman::p_load(tidymodels,tidyverse,ranger,caret,rpart.plot,rpart,
               probably,discrim)

# script to eval rf model from parameter sweep and fs

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
  all_cores <- 4
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
  all_cores <- 10
}

outfile <- file.path(path,'scratch','finalrfcalib_out.txt')
file.remove(outfile)

#all_cores <- parallel::detectCores(logical=TRUE)
cl <- makePSOCKcluster(all_cores,outfile=outfile)
registerDoParallel(cl)

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

# subset features
train <- master %>% 
  filter(set == 'train') %>% 
  select(label,matches(feat_subset)) %>%
  mutate(label=as.character.factor(label),
         label=if_else(label == 1,'delirium','no_delirium'),
         label=as.factor(label))


test <- master %>% 
  filter(set == 'test') %>% 
  select(label,matches(feat_subset)) %>%
  mutate(label=as.character.factor(label),
         label=if_else(label == 1,'delirium','no_delirium'),
         label=as.factor(label))


heldout <- master %>% 
  filter(set == 'heldout_expert') %>% 
  select(label,matches(feat_subset)) %>%
  mutate(label=as.character.factor(label),
         label=if_else(label == 1,'delirium','no_delirium'),
         label=as.factor(label))


metrics <- metric_set(roc_auc,brier_class)

tree_spec <- rand_forest(
  mtry  = params$mtry[1],
  trees = 1500,
  min_n = floor(quantile(1:nrow(train),params$min_node_perc[1]))
) %>%
  set_engine('ranger',
             replace=TRUE, 
             #num.threads=1, # testing this
             verbose=TRUE,
             oob.error=FALSE) %>%
  set_mode('classification')

rf_wf <- workflow() %>%
  add_formula(label ~.) %>%
  add_model(tree_spec)

train_cv <- vfold_cv(train,strata=label,v=10)
ctrl <- control_resamples(save_pred=TRUE)

rf_res <- rf_wf %>%
  fit_resamples(train_cv,metrics=metrics,control=ctrl)

collect_metrics(rf_res)

collect_predictions(rf_res) %>%
  ggplot(aes(.pred_delirium)) +
  geom_histogram(col='black',fill='gray',bins=40) +
  facet_wrap(~label,ncol=1) +
  geom_rug(col='blue',alpha=0.2)

# calibration on cv training
cal_plot_windowed(rf_res,step_size=0.025)

logit_val <- cal_validate_logistic(rf_res,
                                   metrics=metrics,save_pred=TRUE)

collect_metrics(logit_val)

collect_predictions(logit_val) %>%
  filter(.type == 'calibrated') %>%
  cal_plot_windowed(truth=label,estimate=.pred_delirium,step_size=0.025)

beta_val <- cal_validate_beta(rf_res,metrics=metrics,save_pred=TRUE)

collect_metrics(beta_val)

collect_predictions(beta_val) %>%
  filter(.type == 'calibrated') %>%
  cal_plot_windowed(truth=label,estimate=.pred_delirium,step_size=0.025)

fit_cal <- cal_estimate_logistic(rf_res)
rf_fit <- rf_wf %>%
  fit(data=train)

test_pred <- augment(rf_fit,new_data=heldout)
test_pred %>% 
  metrics(label,.pred_delirium)

test_pred_cal <- test_pred %>%
  cal_apply(fit_cal)

test_pred_cal %>%
  select(label,starts_with('.pred_'))

test_pred_cal %>%
  metrics(label,.pred_delirium)

test_pred %>%
  cal_plot_windowed(truth=label, estimate=.pred_delirium,step_size=0.025)

test_pred_cal %>%
  cal_plot_windowed(truth=label, estimate=.pred_delirium,step_size=0.025)




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
  verbose=FALSE,
  probability=TRUE
)

# results on testing set
yhat <- predict(fit,master %>% 
                  filter(set == 'test') %>% 
                  select(label,matches(feat_subset)))$predictions %>%
  as_tibble() %>%
  rename('.pred_neg' = 1,'.pred_pos' = 2) %>%
  mutate(pred=if_else(.pred_pos > 0.5, 1, 0)) %>%
  bind_cols(master %>% 
              filter(set == 'test') %>% 
              mutate(truth=as.integer(as.character.factor(label))) %>%
              select(truth))

conf <- confusionMatrix(table(pull(yhat,truth),pull(yhat,pred)),
                        mode='everything',
                        positive='1')

conf

# calibration on testing
yhat %>%
  mutate(truth = yhat$truth) %>%
  cal_plot_windowed(truth, .pred_pos ,step_size=0.025)
  #cal_plot_breaks(truth, .pred_pos, num_breaks=8)

# results on heldout set
yhat <- predict(fit,master %>% 
                  filter(set == 'heldout_expert') %>% 
                  select(label,matches(feat_subset)))$predictions %>%
  as_tibble() %>%
  rename('.pred_neg' = 1,'.pred_pos' = 2) %>%
  mutate(pred=if_else(.pred_pos > 0.5, 1, 0)) %>%
  bind_cols(master %>% 
              filter(set == 'heldout_expert') %>% 
              mutate(truth=as.integer(as.character.factor(label))) %>%
              select(truth))

conf <- confusionMatrix(table(pull(yhat,truth),pull(yhat,pred)),
                        mode='everything',
                        positive='1')

conf

yhat %>%
  mutate(truth = yhat$truth) %>%
  cal_plot_windowed(truth, .pred_pos ,step_size=0.025)
  #cal_plot_breaks(truth, .pred_pos, num_breaks=8)


train <- master %>% 
  filter(set == 'train') %>% 
  select(label,matches(feat_subset))

set.seed(s1)
fit <- rpart(label ~., 
             data = train)

rpart.plot(fit)
