library(tidyverse)
library(glue)

# script to create a table for training, validation, and testing

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

# create the following
# 1 an unlabeled training set for pretraining/lasso
# 2 an unlabeled validation set for pretraining/lasso
# 3 a labeled test set using icd labels for testing
# 4 a labeled test set using expert labels for self training
# 5 a heldout set from the expert labels for final testing

# filter hc <= 50 or concat hpi_hc <= 100 from full table
# 154267 samples
tbl <- read_rds(file.path(path,'data_in','tbl_final_wperiods.rds')) %>%
  select(id,hpi=history_of_present_illness,hc=hospital_course,
         label,label_icd,hpi_hc) %>%
  filter(nchar(hc) > 50 | str_detect(hc,'see hpi')) %>%
  filter(nchar(hpi_hc) > 100)

cat(glue('\n\nNumber of total samples: {nrow(tbl)}\n\n'))

# pull expert labels
# 5890 samples
tbl_test_expert <- tbl %>%
  filter(label %in% c(0,1)) %>%
  mutate(set='test_expert')

cat(glue('\n\nNumber of expert labeled samples: {nrow(tbl_test_expert)}\n\n'))

# pull 300 samples from expert set for heldout
set.seed(1234)
n_ho0 <- 150
n_ho1 <- 150
tbl_heldout_expert <- bind_rows(tbl_test_expert %>%
                                  filter(label == 0) %>%
                                  sample_n(n_ho0,replace=FALSE) %>%
                                  mutate(set='heldout_expert'),
                                tbl_test_expert %>%
                                  filter(label == 1) %>%
                                  sample_n(n_ho1,replace=FALSE) %>%
                                  mutate(set='heldout_expert'))

cat(glue('\n\nNumber of heldout samples: {nrow(tbl_heldout_expert)}\n\n'))

# remove heldout set from expert set
# 4857 remaining samples
tbl_test_expert <- tbl_test_expert %>%
  anti_join(tbl_heldout_expert,by='id')

cat(glue('\n\nNumber of expert labeled samples after heldout: ',
         '{nrow(tbl_test_expert)}\n\n'))

# remove expert labeled and heldout samples from full table
# 148377 remaining samples
tbl <- tbl %>%
  anti_join(tbl_test_expert,by='id') %>%
  anti_join(tbl_heldout_expert,by='id')

cat(glue('\n\nNumber of samples after removing expert and heldout: ',
         '{nrow(tbl)}\n\n'))

# create testing set from full table. Will be double 10% of the 
# minority class.
# 1648 icd labeled samples    
n <- ceiling(min(table(tbl$label_icd) * 0.1))
tbl_test_icd <- tbl %>%
  filter(label_icd %in% c(0,1)) %>%
  group_by(label_icd) %>%
  sample_n(n,replace=FALSE) %>%
  mutate(set='test_icd')

cat(glue('\n\nNumber of icd labeled samples: {nrow(tbl_test_icd)}\n\n'))

# remove icd labeled samples from full table
# 146729 remaining samp[les
tbl <- tbl %>%
  anti_join(tbl_test_icd,by='id') 

cat(glue('\n\nNumber of samples after removing icd labeled: ',
         '{nrow(tbl)}\n\n'))

# create validation set from full table. Will be 10% of full table
# 1648 validation samples
tbl_val <- tbl %>%
  filter(label_icd %in% c(0,1)) %>%
  group_by(label_icd) %>%
  sample_n(n,replace=FALSE) %>%
  mutate(set='val')

cat(glue('\n\nNumber of validation samples: {nrow(tbl_val)}\n\n'))

# remove validation samples from full table yielding training set
# 145081 training samples
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

