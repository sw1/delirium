pacman::p_load(tidyverse,glue,doParallel)

# script to merge pseudolabels with notes table

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
  all_cores <- 4
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
  all_cores <- 8
}
source(file.path(path,'code','fxns.R'))

outfile <- file.path(path,'scratch','pseudolabel.txt')
if (file.exists(outfile)) file.remove(outfile)

#all_cores <- parallel::detectCores(logical=FALSE)
cl <- makePSOCKcluster(all_cores,outfile=outfile)
registerDoParallel(cl)

tbl <- read_csv(file.path(path,
                          'to_python',
                          'tbl_to_python_expertupdate.csv.gz')) %>%
  mutate(hpi=if_else(is.na(hpi),'',hpi))

# rename testing set to validation
tbl <- tbl %>%
  mutate(set=if_else(set == 'test_expert','val',set))

# moving remaining expert labels not from heldout or test to train
tbl <- tbl %>% 
  mutate(set=if_else(set == 'expert','train',set))

# creating last table with expert labels 1 and all NAs 0. Labels will be
# set as label_icds so longerformer script can be used without modification
tbl_train <- tbl %>%
  filter(set == 'train') %>%
  mutate(label=if_else(label == -1,0,label))

tbl_fullexpert <- tbl %>%
  anti_join(tbl_train,'id') %>% 
  bind_rows(tbl_train) %>%
  select(id,label_fullexpert=label)

# add new label columns
tbl <- tbl %>%
  left_join(tbl_fullexpert,by='id') %>%
  mutate(label_fullexpert=if_else(
    is.na(label_fullexpert),-1,label_fullexpert)) 


pseudos <- read_rds(file.path(path,'data_out',
                              '10_labels_rfst_count_del_full.rds')) 
pseudos$combs <- pseudos$combs %>% 
  mutate(idx=row_number())

combs <- pseudos$combs %>%
  select(thresholds,fracs) %>%
  distinct()

n_votes <- 2

out <- foreach(i=1:nrow(combs),.verbose=TRUE,
               .errorhandling='stop',
               .packages=c('tidyverse','ranger','glue')) %dopar% {
                 
  th <- combs$thresholds[i]
  fr <- combs$fracs[i]
  
  idxs <- pseudos$combs %>% 
    filter(thresholds == th,fracs == fr) %>%
    pull(idx)
  
  labs <- pseudos$out[[idxs[1]]]
  for (j in idxs[-1]){
    labs <- labs %>%
      full_join(pseudos$out[[j]],by='id')
  }
  
  labs <- labs %>% 
    rowwise() %>%
    mutate(vote=case_when(
      sum(c_across(starts_with('label')) == 1) >= n_votes ~ 1,
      sum(c_across(starts_with('label')) == 0) >= n_votes ~ 0,
      TRUE ~ -1)) %>%
    select(id,label=vote) %>%
    rename(!!glue('label_pseudo_th{th*100}_fr{fr*100}') := label) 
  
  return(labs)
  
}

stopCluster(cl)

labs <- out[[1]]
for (i in 2:length(out)){
  labs <- labs %>%
    full_join(out[[i]],by='id')
}

tbl <- tbl %>%
  left_join(labs,by='id') %>%
  mutate(across(everything(),~replace_na(.x, -1)))

tbl <- tbl %>%
  write_csv(file.path(path,'to_python','tbl.csv.gz'))