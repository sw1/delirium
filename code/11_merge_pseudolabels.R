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

# get filenames for pseudolabels
fns <- list.files(file.path(path,'data_in'),
                  pattern='^labels_rfst',full.names=TRUE)

maj_vote <- list()

for (fn in fns){
  
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
