pacman::p_load(tidymodels,tidyverse,doParallel,ranger,glue)

# script to perform entire self training workflow after tables have been
# generated. this allows to subset expert labeled data to eval the amount
# of data needed before a substantial performance drop.

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

fracs <- c(0.75,0.5,0.35,0.2)

fns <- tibble(fn=list.files(file.path(path,'code'))) %>%
  mutate(idx=as.numeric(str_extract(fn,'(\\d+)_.*',group=1))) %>%
  filter(!is.na(idx)) %>%
  arrange(idx) %>%
  filter(idx >= 7 & idx <= 10) 

scripts_dir <- file.path(path,'code_tmp')
out_dir <- file.path(path,'data_tmp','subsamp')

if (file.exists(scripts_dir)){
  file.remove(list.files(scripts_dir,full.names=TRUE))
  file.remove(list.files(out_dir,recursive=TRUE,full.names=TRUE))
}else{
  dir.create(scripts_dir)
  dir.create(out_dir)
  dir.create(file.path(out_dir,'data_in'))
  dir.create(file.path(out_dir,'data_out'))
  dir.create(file.path(out_dir,'data_tmp'))
}

file.copy(file.path(path,'code',fns$fn),scripts_dir)

fns <- fns %>%
  mutate(fn=file.path(scripts_dir,fn))

master <- read_rds(file.path(path,'data_in','06_dat_rf_cv_fs.rds'))

test <- master %>%
  filter(set == 'test_expert') 

heldout <- master %>%
  anti_join(test,by='id')

for (f in fracs){
  
  # subset expert labels
  test_frac <- test %>%
    sample_frac(f)
  
  master_frac <- heldout %>%
    bind_rows(test_frac)
  
  write_rds(master_frac,file.path(out_dir,'data_in','06_dat_rf_cv_fs.rds'))
  
  for (i in sort(fns$idx)){
    fn <- fns %>% filter(idx==i) %>% pull(fn)
    
    cat(glue('\n\nRunning script {fn}\n.'))
    edit_run_script_frac(fn)
  }
  
}

