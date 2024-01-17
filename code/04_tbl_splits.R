library(tidyverse)

z <- function(x) (x-mean(x))/sd(x)

# training data for tokenizer and pretrain
# training and validation data for finetuning
# testing balanced data for final
# haobo for final
# testing + haobo for final
# all 3 testing sets with just hpi

path <- 'D:\\Dropbox\\embeddings\\delirium'

tbl <- read_rds(file.path(path,'data_in','tbl_final_wperiods.rds')) %>%
  select(id,
         hpi=history_of_present_illness,label,hc=hospital_course,
         label,icd_sum) %>%
  mutate(label_icd = if_else(icd_sum>0,1,0)) %>%
  rowwise() %>%
  mutate(hpi_hc = paste(hpi,hc,collapse=' ')) %>%
  ungroup() %>%
  group_by(hpi_hc) %>% 
  mutate(n=n()) %>% 
  ungroup() %>% 
  filter(n == 1 | (n > 1 & !is.na(label))) %>%
  mutate(label = if_else(n > 1,NA,label)) %>%
  distinct(hpi_hc,.keep_all=TRUE) %>%
  select(-n)

set.seed(1234)
p <- 0.1

# haobo test
tbl_test_haobo <- tbl %>%
  filter(label %in% c(0,1)) %>%
  mutate(set='test_haobo')

tbl <- tbl %>%
  anti_join(tbl_test_haobo,by='id')

n <- ceiling(min(table(tbl$label_icd) * p))

# icd test
tbl_test_icd <- tbl %>%
  group_by(label_icd) %>%
  sample_n(n,replace=FALSE) %>%
  mutate(set='test_icd')

tbl <- tbl %>%
  anti_join(tbl_test_icd,by='id') 

# training val
tbl_val <- tbl %>%
  group_by(label_icd) %>%
  sample_n(n,replace=FALSE) %>%
  mutate(set='val')

# training
tbl <- tbl %>%
  anti_join(tbl_val,by='id') %>%
  mutate(set='train')

tbl <- tbl %>%
  bind_rows(tbl_val) %>%
  bind_rows(tbl_test_icd) %>%
  bind_rows(tbl_test_haobo)

write_csv(tbl,
          file.path(path,'to_python','tbl_to_python_updated.csv.gz'),
          col_names=TRUE)

