pacman::p_load(tidyverse,glue)

# script to perform entire self training workflow after tables have been
# generated. this allows to subset expert labeled data to eval the amount
# of data needed before a substantial performance drop.

fracs <- c(0.75,0.5,0.35,0.2)
idx_fracs <- seq_along(fracs)

set.seed(43)
seeds <- sample(1:9999,length(fracs))

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
  all_cores <- 8
  scripts_fn <- 'code_tmp2'
  out_fn <- 'data_tmp2'
  idx_fracs <- rev(idx_fracs)
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
  all_cores <- 16
  scripts_fn <- 'code_tmp1'
  out_fn <- 'data_tmp1'
}
source(file.path(path,'code','fxns.R'))

# filter scripts that involve self training and hence have to be adjusted
fns_raw <- tibble(fn=list.files(file.path(path,'code'))) %>%
  mutate(idx=as.numeric(str_extract(fn,'(\\d+)_.*',group=1))) %>%
  filter(!is.na(idx)) %>%
  arrange(idx) %>%
  filter(idx >= 7 & idx <= 10) 

# create tmp dirs to copy edited scripts and intermediate files
scripts_dir <- file.path(path,scripts_fn)
out_dir <- file.path(path,out_fn)

if (dir.exists(scripts_dir)) unlink(scripts_dir,recursive=TRUE)
if (dir.exists(out_dir)) unlink(out_dir,recursive=TRUE)

dir.create(file.path(out_dir,'scratch'),recursive=TRUE,showWarnings=FALSE)
dir.create(file.path(out_dir,'data_in'),recursive=TRUE,showWarnings=FALSE)
dir.create(file.path(out_dir,'data_out'),recursive=TRUE,showWarnings=FALSE)
dir.create(file.path(out_dir,'data_tmp'),recursive=TRUE,showWarnings=FALSE)
dir.create(file.path(out_dir,'code'),recursive=TRUE,showWarnings=FALSE)
dir.create(scripts_dir,recursive=TRUE,showWarnings=FALSE)

tbl_baseline <- read_csv(
  file.path(path,
            'to_python',
            'tbl_to_python_expertupdate_fullexpert_chunked.csv.gz')
)

# load data file thats used that contains expert labels and can be reused
# throughout loop for subsetting by f
master <- read_rds(file.path(path,'data_in','06_dat_rf_cv_fs.rds'))

test <- master %>%
  filter(set == 'test_expert') 

heldout <- master %>%
  anti_join(test,by='id')

for (i in idx_fracs){
  
  f <- fracs[i]
  
  file.remove(list.files(out_dir,recursive=TRUE,full.names=TRUE))
  file.remove(list.files(scripts_dir,recursive=TRUE,full.names=TRUE))
  
  file.copy(file.path(path,'code',fns_raw$fn),scripts_dir)
  
  fns <- fns_raw %>%
    mutate(fn=file.path(scripts_dir,fn))
  
  file.copy(file.path(path,'code','fxns.R'),
            file.path(out_dir,'code'))
  file.copy(file.path(path,'data_in','notes.csv.gz'),
            file.path(out_dir,'data_in'))
  
  # subset expert labels by f
  set.seed(seeds[i])
  test_frac <- test %>%
    sample_frac(f)
  
  test_anti <- test %>%
    anti_join(test_frac,by='id') %>%
    pull(id)
  
  # filter expert labels from baseline chunked table for later comps
  tbl_baseline %>%
    filter(!(id %in% test_anti)) %>%
    write_csv(
      file.path(path,
                'to_python',
                glue('tbl_to_python_expertupdate_fullexpert_chunked',
                     '_frac{f*100}',
                     '.csv.gz')))
  
  # filter expert labels for all data files that are used and make a copy
  # in tmp dir
  master_frac <- heldout %>%
    bind_rows(test_frac) %>%
    write_rds(file.path(out_dir,'data_in','06_dat_rf_cv_fs.rds'))
  
  read_rds(file.path(path,'data_in','05_full_icd_tbl.rds')) %>%
    filter(!(id %in% test_anti)) %>%
    write_rds(file.path(out_dir,'data_in','05_full_icd_tbl.rds'))
  
  # edit scripts to change path to tmp folder and for file 10 only do
  # thresholding for 0.7 given results
  for (ii in sort(fns$idx)){
    
    fn <- fns %>% filter(idx==ii) %>% pull(fn)
    
    cat(glue('\n\nRunning script {fn}.\n\n'))
    read_lines(fn) %>%
      str_replace(
        pattern='embeddings\\\\\\\\delirium',
        replace=glue('embeddings\\\\\\\\',
                     'delirium\\\\\\\\',
                     '{out_fn}\\\\\\\\')
        ) %>% 
      str_replace(
        pattern='0.6,0.7,0.8,0.9',
        replace='0.7'
        ) %>%
      str_replace(
        pattern='all_cores <- \\d+',
        replace=glue('all_cores <- {all_cores}')
      ) %>%
      str_replace(
        pattern='dopar',
        replace='do'
      ) %>%
      str_replace(
        pattern='num.threads=1',
        replace=glue('num.threads={all_cores}')
      ) %>%
      str_replace(
        pattern='sample_n\\(250\\)',
        replace=glue('sample_n(round(250*{f}))')
      ) %>%
      writeLines(con=fn)
    
    env_tmp <- new.env()
    source(fn,local=env_tmp,echo=FALSE)
    
  }
  
  # essentially run second half of script 11 on new data
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

  # get filenames for pseudolabels from tmp dir
  fns <- list.files(file.path(out_dir,'data_in'),
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
                   "seed{params['seed']}_frac{f*100}.csv.gz")
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
      
      cat(
        glue('\n\n\n Performing majority vote for th={th}, ns={ns}. \n\n\n'))
      
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
                     'th{th}_ns{ns}_frac{f*100}.csv.gz')
      write_csv(tbl_update,file.path(path,'to_python',fn_out))
      
    }
  }
}

if (dir.exists(scripts_dir)) unlink(scripts_dir,recursive=TRUE)
if (dir.exists(out_dir)) unlink(out_dir,recursive=TRUE)
