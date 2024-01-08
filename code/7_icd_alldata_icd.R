library(tidymodels)
library(glmnet)
library(tidyverse)
library(doParallel)
library(vip)
library(rpart.plot)



path <- 'D:\\Dropbox\\embeddings\\delirium'
source(file.path(path,'code','fxns.R'))

ss <- 'icd'

tree_fit <- read_rds(file.path(path,'data_in',sprintf('fit_tree_count_del_%s.rds',ss)))
features <- tree_fit$data %>% select(-id,-label) %>% colnames()

haobo <- read_rds(file.path(path,'data_in',
                            sprintf('full_icd_tbl_%s.rds',
                                    gsub('_','',ss)))) %>%
  select(-icd_codes) %>%
  rename(icd_codes=icd_sub_chapter) 
# filter(!is.na(label))  # turn off for full data

features_icds <- unique(unlist(haobo$icd_codes))
features_icds <- features_icds[features_icds %in% features_icds_tree]
# features_icds <- features_icds[nchar(features_icds) > 0]
icd_mat <- matrix(0,nrow(haobo),length(features_icds))
colnames(icd_mat) <- paste0('icd_',features_icds)
for (i in 1:nrow(haobo)){
  icds <- na.omit(unlist(haobo$icd_codes[i]))
  icds <- icds[icds %in% features_icds_tree]
  if (length(icds) == 0) next
  icds <- paste0('icd_',icds)
  for (j in seq_along(icds)){
    icd_mat[i,icds[j]] <- icd_mat[i,icds[j]] + 1
  }
}

haobo <- haobo %>% select(-icd_codes)

features_service <- unique(unlist(haobo$service))
service_mat <- matrix(0,nrow(haobo),length(features_service))
colnames(service_mat) <- features_service
for (i in 1:nrow(haobo)){
  servs <- unique(unlist(haobo$service[i]))
  service_mat[i,servs] <- 1
}
colnames(service_mat) <- paste0('count_service_',colnames(service_mat))
service_mat <- service_mat[,colnames(service_mat) != 'count_service_other']
colnames(service_mat)[
  colnames(service_mat) == 'count_service_obstetrics/gynecology'
] <- 'count_service_ob'

haobo <- haobo %>% select(-service)

haobo <- haobo %>%                             # added 1/2/2024
  bind_cols(icd_mat) %>%
  bind_cols(service_mat)

# create metadata
haobo <- create_counts(haobo)

haobo_post <- haobo %>%
  select(!starts_with(c('icd_','count_','los_')),contains(features))

write_rds(haobo_post,file.path(path,'data_in',sprintf('alldat_preprocessed_for_pred_%s.rds',ss)))

