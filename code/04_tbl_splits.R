pacman::p_load(tidyverse,glue)

# script to create a table for training, validation, and testing

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

# create the following
# 1 an unlabeled training set for pretraining
# 2 an unlabeled validation set for pretraining
# 3 a heldout set from the expert labels for final testing
# 4 a labeled test set using icd labels for testing, same size as heldout

# filter concat hpi_hc <= 100 from full table
# 154267 samples
tbl <- read_rds(file.path(path,'data_in','03_tbl_final_wperiods.rds')) %>%
  select(id,hpi=history_of_present_illness,hc=hospital_course,
         label,label_icd,hpi_hc) %>%
  filter(nchar(hpi_hc) > 100) 

cat(glue('\n\nNumber of total samples: {nrow(tbl)}\n\n'))

# pull expert labels
# 5890 samples
tbl_test_expert <- tbl %>%
  filter(label %in% c(0,1)) %>%
  mutate(set='test_expert')

cat(glue('\n\nNumber of expert labeled samples: {nrow(tbl_test_expert)}\n\n'))

# remove expert labels full table
# 148377 remaining samples
tbl <- tbl %>%
  anti_join(tbl_test_expert,by='id') 

cat(glue('\n\nNumber of samples after removing expert and heldout: ',
         '{nrow(tbl)}\n\n'))

# pull 10% of samples from expert set for heldout (n=589)
set.seed(241)
tbl_heldout_expert <- tbl_test_expert %>%
  group_by(label) %>%
  sample_frac(0.1,replace=FALSE) %>%
  ungroup() %>%
  mutate(set='heldout_expert')

cat(glue('\n\nNumber of heldout samples: {nrow(tbl_heldout_expert)}\n\n'))

# remove heldout set from expert set
# 5301 remaining expert labeled samples
tbl_test_expert <- tbl_test_expert %>%
  anti_join(tbl_heldout_expert,by='id')

cat(glue('\n\nNumber of expert labeled samples after heldout: ',
         '{nrow(tbl_test_expert)}\n\n'))

# create icd testing set from full table, same size as heldout.
# 589 icd labeled samples    
ids_test_0 <- tbl %>%
  filter(label_icd == 0) %>%
  sample_n(sum(tbl_heldout_expert$label == 0),replace=FALSE) %>%
  pull(id) 
ids_test_1 <- tbl %>%
  filter(label_icd == 1) %>%
  sample_n(sum(tbl_heldout_expert$label == 1),replace=FALSE) %>%
  pull(id) 

tbl_test_icd <- tbl %>%
  filter(id %in% c(ids_test_0,ids_test_1)) %>%
  mutate(set='test_icd')

cat(glue('\n\nNumber of icd labeled samples: {nrow(tbl_test_icd)}\n\n'))

# remove icd labeled samples from full table
# 147788 remaining samples
tbl <- tbl %>%
  anti_join(tbl_test_icd,by='id') 

cat(glue('\n\nNumber of samples after removing icd labeled: ',
         '{nrow(tbl)}\n\n'))

# create validation set from full table. Will be 1% of remaining data.
# This is strictly for pretraining validation and hence can small.
# 740 validation samples
tbl_val <- tbl %>%
  group_by(label_icd) %>%
  sample_frac(0.01,replace=FALSE) %>%
  ungroup() %>%
  mutate(set='val')

cat(glue('\n\nNumber of validation samples: {nrow(tbl_val)}\n\n'))

# remove validation samples from full table yielding training set
# 140399 training samples
tbl <- tbl %>%
  anti_join(tbl_val,by='id') %>%
  mutate(set='train')

cat(glue('\n\nNumber of training samples: {nrow(tbl)}\n\n'))

# combine all tables
tbl <- tbl %>%
  bind_rows(tbl_val) %>%
  bind_rows(tbl_test_icd) %>%
  bind_rows(tbl_test_expert) %>%
  bind_rows(tbl_heldout_expert) 

# relable NAs to -1 for python and save
tbl <- tbl %>%
  mutate(label_icd=if_else(is.na(label_icd),-1,label_icd),
         label=if_else(is.na(label),-1,label))

write_csv(tbl,file.path(path,'to_python','tbl_to_python_expertupdate.csv.gz'))

