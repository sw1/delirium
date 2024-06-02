pacman::p_load(tidyverse,glue)

# script to merge pseudolabels with notes table

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

tbl <- read_csv(file.path(path,
                          'to_python',
                          'tbl_to_python_expertupdate.csv.gz'))


# because notes will be split based on 4096 char limit in longformer, 
# they must be chunked but each chunk still must be associated w/ same label.
# Here, chunk notes into 4096 char overlapping chunks and then 
# reassociate the label. Then filter any chunks less than 101 chars per
# script 04.

tbl <- tbl %>% 
  rowwise() %>%
  mutate(hpi_hc=if_else(nchar(hpi_hc) > 4096,
                        chunk(hpi_hc),
                        list(hpi_hc))) %>%
  ungroup() %>%
  unnest(hpi_hc) %>%
  filter(nchar(hpi_hc) > 100)

# shouldnt need such a large expert test set (5580) and having some
# true labels in longformer training would be helpful so going to
# plan for an expert test set of 1000 (500/500) and will rename
# remaining examples as train_expert

set.seed(2)
ids_test_expert <- tbl %>% 
  filter(set == 'test_expert') %>%
  select(id,label) %>%
  distinct() %>%
  group_by(label) %>%
  sample_n(500) %>%
  pull(id)

tbl <- tbl %>%
  mutate(set=if_else(set == 'test_expert','train_expert',set),
         set=if_else(id %in% ids_test_expert,'test_expert',set))

write_csv(tbl,
          file.path(path,
                    'to_python',
                    'tbl_to_python_expertupdate_chunked.csv.gz'))

# creating a table with only expert labels for extra longformer eval
tbl_onlyexpert <- tbl %>%
  filter(label != -1) %>%
  mutate(set=if_else(set == 'heldout_expert','heldout_expert','train'))

# create balanced val set from expert labeled training set, 10%
n_val <- tbl_onlyexpert %>% 
  select(id,label) %>% 
  distinct() %>%
  select(label) %>%
  group_by(label) %>%
  reframe(n=n()) %>%
  filter(n == min(n)) %>%
  mutate(n=floor(n*0.05)) %>%
  pull(n)
      
tbl_onlyexpert_val <- tbl_onlyexpert %>%
  filter(set == 'train') %>%
  select(id,label) %>%
  distinct() %>%
  group_by(label) %>%
  sample_n(n_val) %>%
  pull(id)

tbl_onlyexpert <- tbl_onlyexpert %>%
  mutate(set=if_else(id %in% tbl_onlyexpert_val,'val',set))

# adding back the icd test set and also the expert test set which will also
# be in the training set. Only adding the expert test set so the longformer
# script wont have to be edited further. Gave samples unique ids starting
# with 9990000 to avoid any overlap with original ids.
tbl_onlyexpert <- tbl_onlyexpert %>%
  bind_rows(tbl %>% filter(set == 'test_icd')) %>%
  bind_rows(tbl %>% 
              filter(set == 'test_expert') %>%
              mutate(id = row_number() + 9990000)) 

write_csv(tbl_onlyexpert,
          file.path(path,
                    'to_python',
                    'tbl_to_python_expertupdate_onlyexpert_chunked.csv.gz'))

write_csv(tbl_onlyexpert %>%
            mutate(hpi_hc=glue('{hpi} {hc}')) %>%
            distinct(),
          file.path(path,
                    'to_python',
                    'tbl_to_python_expertupdate_onlyexpert.csv.gz'))

# creating last table with expert labels 1 and all NAs 0. Labels will be
# set as label_icds so longerformer script can be used without modification.
# Because labels change, will reset val set so its balanced.
tbl_fullexpert <- tbl %>%
  select(-label) %>%
  left_join(read_csv(file.path(path,'data_in','notes.csv.gz')) %>%
              select(id=rdr_id,label=postop_delirium_yn) %>%
              distinct(),by='id') %>%
  mutate(label_icd=label,
         set=if_else(set != 'heldout_expert','train',set))

n_val <- tbl_fullexpert %>% 
  filter(set == 'train') %>%
  select(id,label) %>% 
  distinct() %>%
  select(label) %>%
  group_by(label) %>%
  reframe(n=n()) %>%
  filter(n == min(n)) %>%
  mutate(n=floor(n*0.05)) %>%
  pull(n)

tbl_fullexpert_val <- tbl_fullexpert %>%
  filter(set == 'train') %>%
  select(id,label) %>%
  distinct() %>%
  group_by(label) %>%
  sample_n(n_val) %>%
  pull(id)

tbl_fullexpert <- tbl_fullexpert %>%
  mutate(set=if_else(id %in% tbl_fullexpert_val,'val',set))

# reinstering test_expert with unique ids just so longerformer script
# runs without modification. will be duplicated ids/examples.
tbl_fullexpert <- tbl_fullexpert %>%
  bind_rows(tbl %>% 
              filter(set == 'test_icd') %>%
              mutate(id = row_number() + 8880000)) %>%
  bind_rows(tbl %>% 
              filter(set == 'test_expert') %>%
              mutate(id = row_number() + 9990000)) 

write_csv(tbl_fullexpert,
          file.path(path,
                    'to_python',
                    'tbl_to_python_expertupdate_fullexpert_chunked.csv.gz'))

write_csv(tbl_fullexpert %>%
            mutate(hpi_hc=glue('{hpi} {hc}')) %>%
            distinct(),
          file.path(path,
                    'to_python',
                    'tbl_to_python_expertupdate_fullexpert.csv.gz'))

# get filenames for pseudolabels
fns <- list.files(file.path(path,'data_in'),
                  pattern='^labels_rfst',full.names=TRUE)

maj_vote <- list()

for (fn in fns){
  
  fn_short <- str_extract(fn,'(labels_rfst.*.csv.gz)',group=1)
  cat(glue('\n\n\n Updating labels for {fn_short}. \n\n\n'))
  
  # get parameters from filenames
  params <- str_replace(fn,'^.*\\/','')
  params <- str_replace(params,'\\.csv\\.gz','')
  params <- str_split(params,'_')[[1]]
  params <- params[str_detect(params,'[[:digit:]]')]
  params <- str_match(params,'(\\D+)(\\d+\\.?\\d*)')
  
  params_tmp <- params[,3]
  names(params_tmp) <- params[,2]
  params <- params_tmp
  
  params['th'] <- as.character(as.numeric(params['th'])*100)

  new_labels <- read_csv(fn)
  
  tbl_update <- tbl %>%
    left_join(new_labels %>%
                rename(label_pseudo=label),
              by='id') %>%
    mutate(label_pseudo=if_else(is.na(label_pseudo),-1,label_pseudo)) 
  
  fn_out <- glue('tbl_to_python_expertupdate_chunked_rfst_',
                 "th{params['th']}_ns{params['ns']}_",
                 "seed{params['seed']}.csv.gz")
  write_csv(tbl_update,file.path(path,'to_python',fn_out))

  # try to append labels for seed s to df with df of labels from other seeds
  # but if df does not yet exist, create it
  maj_vote[[params['th']]][[params['ns']]] <- try(
    maj_vote[[params['th']]][[params['ns']]] %>%
      bind_cols(tbl_update %>% select(label_pseudo)))
  if (class(maj_vote[[params['th']]][[params['ns']]])[1] == 'try-error'){
    maj_vote[[params['th']]][[params['ns']]] <- tbl_update %>% 
      select(label_pseudo)
  }
  
}

# get majority vote for each threshold and feature set
for (th in names(maj_vote)){
  
  for (ns in names(maj_vote[[th]])){
  
    cat(glue('\n\n\n Performing majority vote for th={th}, ns={ns}. \n\n\n'))

    # skip if only one seed
    if (ncol(maj_vote[[th]][[ns]]) < 3) next
    
    # get majority vote across seeds based on median
    # if median does not result in an integer (hence a tie),
    # then set label as -1 (indicator for no label),
    # then merge with original data. Did it the way below
    # since its faster than median(c_across())

    vote <- maj_vote[[th]][[ns]] %>% 
      select(starts_with('label')) %>%
      as.matrix() 
    
    vote <- apply(vote,1,function(x){
      x <- median(x,na.rm=TRUE)
      x <- if (x %% 1 == 0.5) -1 else x
      return(x)
    })
    
    vote <- tibble(label_pseudo=vote)
    
    tbl_update <- tbl %>%
      bind_cols(vote)
    
    fn_out <- glue('tbl_to_python_expertupdate_chunked_rfst_majvote_',
                   'th{th}_ns{ns}.csv.gz')
    write_csv(tbl_update,file.path(path,'to_python',fn_out))
    
  }
}
