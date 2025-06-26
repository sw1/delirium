pacman::p_load(tidyverse,glue,gtsummary,flextable,icd.data,tidymodels,
               rpart,rpart.plot,officer,gridExtra,stm,LDAvis)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

K <- 30

params <- tibble(th = c('70','80','90','90','90'),
                 pl = c('1','1','1','2','1'),
                 mod = c('lf','lf','lf','lf','lam')) 


weights <- tibble()
for (i in 1:nrow(params)){
  
  th <- as.numeric(params$th[i])
  pl <- as.numeric(params$pl[i])
  mod <- params$mod[i]

  dat <- read_rds(file.path(path,'data_out',
                            glue('stm_th{th}_pl{pl}_mod{mod}.rds')))
  
  processed <- dat$processed
  docs <- dat$docs
  tm <- dat$tm
  K <- tm$settings$dim$K
  rm(dat)
  
  for (f in c('preds','tp','tn','fp','fn')){
    
    formula <- as.formula(glue('1:K ~ {f}'))
    
    set.seed(123412)
    eff <- estimateEffect(formula,tm,
                          meta=processed$meta,
                          uncertainty='Global')
    
    res <- summary(eff)
    
    weights <- weights %>%
      bind_rows(tibble(w=sapply(res[[3]], function(x) x[2,1])) %>%
                  mutate(th=th,
                         pl=pl,
                         mod=mod,
                         feature=f,
                         K=row_number()))
    
  }
}


write_csv(weights,file.path(path,'data_out','stm_weights.csv.gz'))
  



