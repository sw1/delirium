library(tidyverse)

path <- 'D:\\Dropbox\\embeddings\\delirium'

tbl <- read_csv(file.path(path,'to_python','tbl_to_python_updated.csv.gz'))
# all have same test ids, so rf icd
test_ids <- read_csv(file.path(path,'data_out','test_set_rf_icd.csv.gz')) 

tbl_heldout <- tbl %>%
  right_join(test_ids,by='id')

tbl_update <- tbl %>% 
  filter(set != 'test_haobo') %>%
  bind_rows(tbl_heldout)

write_csv(tbl_update,
          file.path(path,'to_python','tbl_to_python_updated_treeheldout.csv.gz'),
          col_names=TRUE)



