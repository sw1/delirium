library(tidyverse)

ds_type <- c('count_del','binary_del')

path <- 'D:\\Dropbox\\embeddings\\delirium'

tbl <- read_csv(file.path(path,'to_python','tbl_to_python_updated.csv.gz'))

ds <- ds_type[1]
ss <- 'icd'

# new_labels <- read_csv(file.path(path,paste0('labels_sptree_',ds,'.csv'))) %>%
#   distinct()

new_labels <- read_csv(file.path(path,'data_in',paste0('labels_sptree_',ds,sprintf('_%s.csv.gz',ss)))) %>%
  distinct()

tbl_update <- new_labels %>% 
  select(id,label_tree) %>%
  mutate(label_icd = label_tree) %>%
  select(-label_tree) %>%
  left_join(tbl %>% select(-label_icd),by='id') 

tbl_update %>% group_by(set,label_icd) %>% summarize(n())

# write_csv(tbl_update,
#           file.path(path,sprintf('tbl_to_python_231229_%s.csv.gz',ds)),
#           col_names=TRUE)

write_csv(tbl_update,
          file.path(path,'to_python',sprintf('tbl_to_python_updated_%s_%s.csv.gz',ds,ss)),
          col_names=TRUE)

