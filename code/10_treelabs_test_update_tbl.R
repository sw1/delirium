library(tidyverse)

path <- 'D:\\Dropbox\\embeddings\\delirium'

exts <- c('icd','major','sub_chapter')
for (ext in exts){
  tbl <- read_csv(file.path(path,'to_python',sprintf('tbl_to_python_updated_count_del_%s.csv.gz',ext)))
  test_ids <- read_csv(file.path(path,'data_out',sprintf('test_set_sptree_%s.csv.gz',ext)))
  
  tbl_update <- tbl %>% 
    right_join(test_ids,by='id') %>%
    mutate(set='test_tree')
  
  write_csv(tbl_update,
            file.path(path,'to_python',sprintf('tbl_to_python_updated_treelabs_%s.csv.gz',ext)),
            col_names=TRUE)
}