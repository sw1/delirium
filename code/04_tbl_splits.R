pacman::p_load(tidyverse,glue)

# script to create a table for training, validation, and testing

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

# create the following
# 1 an unlabeled training set for pretraining
# 2 an unlabeled validation set for pretraining
# 3 a heldout set from the expert labels for final testing
# 4 a labeled test set using icd labels for testing, same size as heldout

# filter concat hpi_hc <= 100 from full table
# 154267 samples
# tbl <- read_rds(file.path(path,'data_out','03_tbl_final_wperiods.rds')) %>%
#   select(id,hpi=history_of_present_illness,hc=hospital_course,
#          label,label_icd,hpi_hc) %>%
#   filter(nchar(hpi_hc) > 100) 

# for updated heldout without overlap per brett
# filter concat hpi_hc <= 100 from full table and
# remove starting NAs
# 154267 samples
tbl <- read_rds(file.path(path,'data_out','03_tbl_final_wperiods.rds')) %>%
select(id,mrn,admission_date,
       hpi=history_of_present_illness,hc=hospital_course,
       label,label_icd,hpi_hc) %>%
  filter(nchar(hpi_hc) > 100) 

n_start_rows <- nrow(tbl)

cat(glue('\n\nNumber of total samples: {nrow(tbl)}\n\n'))

# pull expert labels
# 5890 samples
tbl_expert <- tbl %>%
  filter(label %in% c(0,1)) %>%
  mutate(set='expert')

cat(glue('\n\nNumber of expert labeled samples: {nrow(tbl_expert)}\n\n'))

# remove expert labels from full table
# 148377 remaining samples
tbl <- tbl %>%
  anti_join(tbl_expert,by='id') 

cat(glue('\n\nNumber of samples after removing expert: ',
         '{nrow(tbl)}\n\n'))

# for updated heldout without overlap per brett
# pull 20% for testing and heldout sets but
# samples are most recent expert labeled samples and do not 
# have patient overlap with training set
tbl_testing_heldout_expert <- bind_rows(
  tbl_expert %>%
    filter(label == 1,
           !(mrn %in% tbl$mrn)) %>%
    arrange(desc(admission_date)) %>%
    slice_head(n=ceiling(0.2*table(tbl_expert$label)[2])),
  tbl_expert %>%
    filter(label == 0,
           !(mrn %in% tbl$mrn)) %>%
    arrange(desc(admission_date)) %>%
    slice_head(n=ceiling(0.2*table(tbl_expert$label)[1]))
) %>%
  ungroup() %>%
  mutate(set='heldout_expert') %>%
  select(-mrn,admission_date)

# bootstrap nchar average and sd after 50 iters
set.seed(241)
r <- 50
text_lens <- nchar(tbl_testing_heldout_expert$hpi_hc)
err <- sapply(seq_len(r), function(i) {
  s <- sample(text_lens,round(nrow(tbl_testing_heldout_expert)/2),replace=TRUE)
  mean(s)
  })

set.seed(3894)
counter <- 1
while(TRUE){
  tbl_tmp <- tbl_testing_heldout_expert
  
  tbl_heldout_expert <- tbl_tmp %>% 
    group_by(label) %>%
    sample_frac(0.5)
  
  tbl_test_expert <- tbl_tmp %>% 
    anti_join(tbl_heldout_expert,by='id')
  
  u_len <- mean(nchar(tbl_heldout_expert$hpi_hc))
  if (u_len > mean(err) + sd(err) | u_len < mean(err) - sd(err)){
    next
  }
  
  u_len <- mean(nchar(tbl_test_expert$hpi_hc))
  if (u_len > mean(err) + sd(err) | u_len < mean(err) - sd(err)){
    next
  }else{
    break
  }
  counter <- counter + 1
}

cat(glue('\n\nPerformed {counter} iterations to find heldout and test sets\n'))
cat(glue('\n\nNumber of test samples: {nrow(tbl_test_expert)}\n'))
cat(glue('\n\nNumber of heldout samples: {nrow(tbl_heldout_expert)}\n\n'))


# remove heldout set from expert set
# 5300 remaining expert labeled samples
tbl_expert <- tbl_expert %>%
  anti_join(tbl_heldout_expert,by='id') %>%
  anti_join(tbl_test_expert,by='id')

cat(glue('\n\nNumber of remaining expert samples after ',
         'removing test and heldout sets: ',
         '{nrow(tbl_expert)}\n\n'))

# pull 10% of samples from expert set for heldout (n=589)
# set.seed(241)
# tbl_heldout_expert <- tbl_test_expert %>%
#   group_by(label) %>%
#   sample_frac(0.1,replace=FALSE) %>%
#   ungroup() %>%
#   mutate(set='heldout_expert')
# cat(glue('\n\nNumber of heldout samples: {nrow(tbl_heldout_expert)}\n\n'))

# combine all tables and relable NAs to -1 for python and save
tbl <- tbl %>%
  mutate(set='train') %>%
  bind_rows(tbl_expert %>% mutate(set='expert')) %>%
  bind_rows(tbl_test_expert %>% mutate(set='test_expert')) %>%
  bind_rows(tbl_heldout_expert %>% mutate(set='heldout_expert')) %>%
  select(-mrn,-admission_date) %>%
  mutate(label_icd=if_else(is.na(label_icd),-1,label_icd),
         label=if_else(is.na(label),-1,label))

cat(glue('\n\nNumber of samples in full table: ',
         '{nrow(tbl)}\n\n'))

cat('\nSet sizes:')
print(table(tbl$set))

cat(glue('\n\nAll unique ids: {nrow(tbl) == length(unique(tbl$id))}\n'))
cat(glue('\n\nNumber of rows correct: {n_start_rows == nrow(tbl)}\n\n'))

write_csv(tbl,file.path(path,'to_python','tbl_to_python_expertupdate.csv.gz'))

