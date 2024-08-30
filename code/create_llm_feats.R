create_tbl <- function(){
  
  tbl <- read_csv(file.path(path,'to_python','tbl.csv.gz'))
  feats <- read_rds(file.path(path,'data_out','09_alldat_preprocessed_for_pred.rds'))
  
  runs <- list.files(file.path(path,'from_python'),pattern='^fkw.*cw5.*(unlab|exp)')
  preds <- tibble()
  for (run in runs){
    params <- str_match_all(
      run,'th(\\d+).*pl(\\d+).*labpseudo_([a-z]+)[[:punct:]](.*)')[[1]][,-1] %>%
      tibble() %>%
      rename(value=1) %>%
      mutate(p=c('th','pl','lab','mod')) %>%
      pivot_wider(names_from='p',values_from='value') %>%
      mutate(mod=if_else(str_detect(mod,'llama'),'lam','lf'))
    
    preds <- preds %>%
      bind_rows(read_csv(file.path(path,'from_python',run)) %>%
                  group_by(id) %>%
                  filter(i == max(i)) %>%
                  ungroup() %>%
                  select(id,labels,preds) %>%
                  bind_cols(params))
  }
  
  
  preds <- preds %>%
    mutate(preds_type=if_else(lab == 'expert',
                              case_when(
                                labels == 1 & preds == 1 ~ 'tp',
                                labels == 0 & preds == 1 ~ 'fp',
                                labels == 1 & preds == 0 ~ 'fn',
                                labels == 0 & preds == 0 ~ 'tn'),
                              case_when(
                                preds == 1 ~ 'p',
                                preds == 0 ~ 'n'))) %>%
    left_join(feats %>%
                select(-set,-label),
              by='id') %>%
    select(-id) %>%
    mutate(across(where(is.numeric), ~ rescale(.x,to=c(0,10)))) 
  
  return (preds)
}





create_fig <- function(tbl,param,label_type,n_feats=25){
  
  param <- sym(param)
  
  icd9_lookup <- tibble(icd9cm_hierarchy) %>% mutate_all(str_to_lower)
  icd10_lookup <- tibble(icd10cm2016) %>%  mutate_all(str_to_lower)
  
  if (param == 'th'){
    lvl_order <- rev(c('90','80','70'))
    tbl <- tbl %>%
      filter(pl == 1,
             mod == 'lf') %>%
      mutate(th = factor(th,levels=lvl_order,ordered=TRUE))
    bar_label <- 'Pseudo-Label Threshold'
  }else if (param == 'pl'){
    lvl_order <- c('FT','FT + PT','FT + PT + T')
    tbl <- tbl %>%
      filter(th == 90,
             pl != 3,
             mod == 'lf') %>%
      mutate(pl = case_when(
        pl == 1 ~ 'FT',
        pl == 2 ~ 'FT + PT'
      ),
      pl = factor(pl,levels=lvl_order,ordered=TRUE))
    bar_label <- 'Pipeline'
  }else{
    lvl_order <- c('Longformer','Llama')
    tbl <- tbl %>%
      filter(pl == 1,
             th == 90) %>%
      mutate(mod = if_else(mod == 'lam','Llama','Longformer'),
             mod = factor(mod,levels=lvl_order,ordered=TRUE))
    bar_label <- 'LLM'
  }
  
  if (label_type == 'expert'){
    tbl <- tbl %>%
      filter(lab == 'expert')
  }else{
    tbl <- tbl %>%
      filter(lab == 'unlabeled')
  }
  
  tbl <- tbl %>%
    select(!!param,preds_type:last_col()) %>%
    group_by(!!param,preds_type) %>%
    summarise_all(mean) %>%
    pivot_longer(cols=-c(!!param,preds_type),
                 names_to='feature',values_to='mean') %>%
    ungroup()
  
  if (label_type == 'expert'){
    tbl <- tbl %>% 
      mutate(preds_type = case_when(
        preds_type == 'fn' ~ 'False Negatives',
        preds_type == 'fp' ~ 'False Positives',
        preds_type == 'tp' ~ 'True Positives',
        preds_type == 'tn' ~ 'True Negatives'))
  }else{
    tbl <- tbl %>% 
      mutate(preds_type = if_else(preds_type == 'p',
                                  'Positive Predictions',
                                  'Negative Predictions'))
  }
  
  if (label_type == 'expert'){
    feature_subset <- tbl %>%
      mutate(preds_type = if_else(preds_type %in% c('False Negatives',
                                                    'False Positives'),
                                  'miss','hit')) 
  }else{
    feature_subset <- tbl 
  }
  feature_subset <- feature_subset %>%
    group_by(feature,preds_type) %>%
    mutate(sd=abs(diff(range(mean)))) %>%
    group_by(feature) %>%
    mutate(sd=max(sd)) %>%
    ungroup() %>%
    distinct(feature,sd) %>%
    arrange(desc(sd)) %>%
    slice_head(n=n_feats) %>%
    pull(feature)
  
  tbl <- tbl %>%
    filter(feature %in% feature_subset) %>%
    left_join(icd9_lookup %>% 
                select(feature=code,short_desc) %>%
                bind_rows(icd10_lookup %>% select(feature=code,short_desc)) %>%
                distinct() %>%
                mutate(feature = paste0('icd_',feature)),
              by='feature') %>%
    mutate(feature=case_when(
      !is.na(short_desc) ~ paste0('ICD: ',short_desc),
      str_detect(feature,'count_service') ~ glue('service: {str_match(feature,"service_(.*)")[,2]}'),
      feature == 'count_home' ~ 'discharged home',
      feature == 'num_meds' ~ 'number of admit meds',
      feature == 'count_psych_med' ~ 'number of admit psych meds',
      feature == 'discharge_date' ~ 'discharge date',
      feature == 'count_conf_ms' ~ 'discharge mental status: confused',
      feature == 'count_tox' ~ 'hospital course: toxic',
      feature == 'count_hep' ~ 'hospital course: liver disease terms',
      feature == 'count_enceph' ~ 'hospital course: enceph',
      feature == 'count_del_dd' ~ 'discharge diagnosis: deliri',
      feature == 'count_del' ~ 'hospital course: deliri',
      feature == 'count_conf_hc' ~ 'hospital course: confusion terms',
      feature == 'count_geri' ~ 'hospital course: geriatr',
      feature == 'count_exf' ~ 'discharged with services',
      feature == 'july_august' ~ 'admitted july or august',
      TRUE ~ feature)) 
  
  feature_order <- tbl %>%
    filter(!!param == rev(unique(!!param))[1]) %>%
    group_by(feature) %>%
    mutate(mean=mean/sum(mean))
  if (label_type == 'expert'){
    feature_order <- feature_order %>%
      filter(preds_type == 'False Negatives') 
  }else{
    feature_order <- feature_order %>%
      filter(preds_type == 'Positive Predictions') 
  }
  feature_order <- feature_order %>%
    arrange(desc(mean)) %>%
    pull(feature)
  
  fig <- tbl %>%
    mutate(feature=factor(feature,levels=feature_order,ordered=TRUE),
           !!param := factor(!!param,levels=lvl_order,ordered=TRUE),
           param := !!param) %>%
    ggplot(aes(x=feature,y=mean,group=preds_type,fill=preds_type)) +
    geom_col(color='black',width=.7,position='fill',alpha=.2) +
    facet_grid(~ param) +
    coord_flip() +
    theme_classic() +
    theme(legend.position = 'bottom',
          plot.title = element_text(face = "plain", size = 10)) +
    labs(x='',
         title=bar_label,
         y='Scaled Relative Means',
         fill='') +
    scale_fill_brewer(type='qual',palette='Set1') +
    scale_y_continuous(breaks = c(0, 1/4, 1/2, 3/4, 1), 
                       labels = scales::percent(c(0, 1/4, 1/2, 3/4, 1))) +
    geom_hline(yintercept = c(1/4,1/2,3/4),linetype=3)
  
  return(fig)
  
}

tbl <- create_tbl()

# for a given feature, FN are more common to occur for threshold 70...
# for a different feature, FP are more common to occur for threshold 90...

(fig <- create_fig(tbl,param='th',label_type='expert'))
ggsave(plot = fig,
       file.path(path,'figs','feats_th.png'),width=12,height=5,dpi = 300)


(fig <- create_fig(tbl,param='pl',label_type='expert'))
ggsave(plot = fig,
       file.path(path,'figs','feats_pl.png'),width=12,height=5,dpi = 300)


(fig <- create_fig(tbl,param='mod',label_type='expert'))
ggsave(plot = fig,
       file.path(path,'figs','feats_mod.png'),width=12,height=5,dpi = 300)


(fig <- create_fig(tbl,param='th',label_type='unlabeled'))
ggsave(plot = fig,
       file.path(path,'figs','feats_th_unlabeled.png'),width=7,height=5,dpi = 300)


(fig <- create_fig(tbl,param='pl',label_type='unlabeled'))
ggsave(plot = fig,
       file.path(path,'figs','feats_pl_unlabeled.png'),width=7,height=5,dpi = 300)


(fig <- create_fig(tbl,param='mod',label_type='unlabeled'))
ggsave(plot = fig,
       file.path(path,'figs','feats_mod_unlabeled.png'),width=7,height=5,dpi = 300)

