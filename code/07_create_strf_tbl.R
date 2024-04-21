library(tidyverse)
library(tidymodels)
library(glue)

# script to build full table using both labeled and unlabeled data for
# prediction during self training

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

gc()

master <- read_rds(file.path(path,'data_in','full_icd_tbl.rds'))

# load features used during rf training to filter unused features
features <- training(
  read_rds(file.path(path,'data_in','fit_st_rf.rds'))$split) %>%
  select(-label) %>%
  colnames() 

features_train <- gsub('icd_','',
                 features[!str_detect(features,'^count_|^los_')])

features_icds <- unique(unlist(master %>% pull(icd_codes)))
features_icds <- features_icds[features_icds %in% features_train]

# create icd indicators
icd_mat <- matrix(0,nrow(master),length(features_icds))
colnames(icd_mat) <- paste0('icd_',features_icds)
for (i in 1:nrow(master)){
  icds <- na.omit(unlist(master$icd_codes[i]))
  icds <- icds[icds %in% features_train]
  if (length(icds) == 0) next
  icds <- paste0('icd_',icds)
  for (j in seq_along(icds)){
    icd_mat[i,icds[j]] <- icd_mat[i,icds[j]] + 1
  }
}

master <- master %>% 
  select(-icd_codes,-icd_codes_del)

# create service indicators
features_service <- unique(unlist(master$service))
service_mat <- matrix(0,nrow(master),length(features_service))
colnames(service_mat) <- features_service
for (i in 1:nrow(master)){
  servs <- unique(unlist(master$service[i]))
  service_mat[i,servs] <- 1
}
colnames(service_mat) <- paste0('count_service_',colnames(service_mat))
service_mat <- service_mat[,colnames(service_mat) != 'count_service_other']
colnames(service_mat)[
  colnames(service_mat) == 'count_service_obstetrics/gynecology'
] <- 'count_service_ob'

# merge all tables
master <- master %>% 
  select(-service) %>%     
  left_join(read_csv(file.path(path,'data_in','notes.csv.gz')) %>%
              select(id=rdr_id,note=note_txt),
            by='id') %>%
  bind_cols(icd_mat) %>%
  bind_cols(service_mat)

rm(list=c('icd_mat','service_mat','features_icds','features_service'))
  
# create metadata
master <- create_counts(master)

# filter unused features using feature list from above
master <- master %>%
  select(!starts_with(c('icd_','count_','los_')),matches(features))

write_rds(master,file.path(path,
                          'data_in',
                          'alldat_preprocessed_for_pred.rds'))



