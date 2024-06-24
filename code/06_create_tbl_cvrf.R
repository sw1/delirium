pacman::p_load(tidymodels,tidyverse,doParallel,vip,icd.data,ranger,glue)

# script to perform initial random forest for self training on expert
# labeled notes only for parameter cross validation

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

col_filter <- 50 # filter  features with less than this many sample occurrences

master <- read_rds(file.path(path,'data_out','05_full_icd_tbl.rds')) %>%
  filter(!is.na(label))  # turn off for full data, note heldout removed below

# create icd indicators
features_icds <- unique(unlist(master$icd_codes))
features_icds <- features_icds[nchar(features_icds) > 0]
icd_mat <- matrix(0,nrow(master),length(features_icds))
colnames(icd_mat) <- paste0('icd_',features_icds)
for (i in 1:nrow(master)){
  icds <- na.exclude(unlist(master$icd_codes[i]))
  icds <- icds[nchar(icds) > 0]
  if (length(icds) == 0) next
  icds <- paste0('icd_',icds)
  for (j in seq_along(icds)){
    icd_mat[i,icds[j]] <- icd_mat[i,icds[j]] + 1
  }
}
icd_mat <- icd_mat[,colSums(icd_mat) >= col_filter]

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
service_mat <- service_mat[,colSums(service_mat) >= col_filter]

# merge feature tables and add raw notes for problem list and hashes
master <- master %>% 
  select(-service,-icd_codes,-icd_codes_del) %>%
  left_join(read_csv(file.path(path,'data_in','notes.csv.gz')) %>%
              select(id=rdr_id,note=note_txt),
            by='id') %>%
  bind_cols(icd_mat) %>%
  bind_cols(service_mat)

# create metadata
master <- create_counts(master,nurse=FALSE)

# select features to be used for rfst
master <- master %>%
  select(id,set,label,
         los,sex,age,
         num_meds,num_allergies,len_pmhx,
         year,
         month_academic,discharge_date,
         starts_with('july'),
         starts_with('covid'),
         starts_with('icd_'),starts_with('count_')) %>%
  select(-icd_sum) 

# replace missing values with 0 and normalize features
master <- master %>%
  mutate(across(everything(),~replace_na(.x, 0)))

cat(glue('\nNumber of features: {ncol(master)-3}.\n\n'))

write_rds(master,file.path(path,'data_out','06_dat_rf_cv_fs.rds'))



  

