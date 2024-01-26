library(tidyverse)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}

tbl <- read_csv(file.path(path,'to_python','tbl_to_python_updated.csv.gz'))
tbl_chunked <- read_csv(file.path(path,
                                  'to_python',
                                  'tbl_to_python_updated_chunked.csv.gz'))

test_ids <- read_csv(file.path(path,'data_out','heldout_tree_set.csv.gz')) %>%
  select(id)

tbl_heldout <- tbl %>%
  right_join(test_ids,by='id')

tbl_update <- tbl %>% 
  filter(set != 'test_haobo') %>%
  bind_rows(tbl_heldout)

write_csv(tbl_update,
          file.path(path,'to_python',
                    'tbl_to_python_updated_treeheldout.csv.gz'),
          col_names=TRUE)

tbl_heldout <- tbl_chunked %>%
  right_join(test_ids,by='id')

tbl_update <- tbl_chunked %>% 
  filter(set != 'test_haobo') %>%
  bind_rows(tbl_heldout)

write_csv(tbl_update,
          file.path(path,'to_python',
                    'tbl_to_python_updated_chunked_treeheldout.csv.gz'),
          col_names=TRUE)

