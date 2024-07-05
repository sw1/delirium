pacman::p_load(tidyverse,textstem,tidytext,lubridate,tm,glue,doParallel)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
  all_cores <- 4
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
  all_cores <- 8
}
source(file.path(path,'code','fxns.R'))

cl <- makeCluster(all_cores)
registerDoParallel(cl)

tbl <- read_csv(file.path(path,'to_python','tbl.csv.gz')) 

years <- list.files(file.path(path,'data_tmp'),
                    pattern='notes_proc_[0-9]+\\.rds')
years <- unique(str_extract(years,'[0-9]+'))

notes <- foreach(i = seq_along(years), 
                 .combine=rbind, 
                 .export=c('years'),
                 .packages=c('tidyverse','glue')) %dopar% {
                   
                   read_rds(
                     file.path(path,'data_tmp',
                               glue('notes_proc_{years[i]}.rds'))) %>%
                     group_by(id) %>%
                     arrange(note_date) %>%
                     reframe(note=paste(note,collapse=' '),
                             length=nchar(note),
                             n_tokens=length(str_split(note,' ')[[1]])) 
                 }

stopCluster(cl)

notes <- notes %>%
  right_join(tbl,by='id')

notes %>%
  mutate(hpi_hc = note) %>%
  select(all_of(colnames(tbl))) %>%
  write_csv(file.path(path,'to_python','tbl_allnotes.csv.gz'))

notes <- notes %>%
  mutate(hpi_hc = glue('{hc} {note}')) %>%
  filter(nchar(hpi_hc) > 100)
  
notes %>%
  filter(n_tokens <= 8192) %>%
  select(all_of(colnames(tbl))) %>%
  write_csv(file.path(path,'to_python','tbl_allnotes_trimmed.csv.gz'))

notes %>%
  select(all_of(colnames(tbl))) %>%
  rowwise() %>%
  mutate(hpi_hc=if_else(nchar(hpi_hc) > 8192,
                          chunk(hpi_hc),
                          list(hpi_hc))) %>%
  ungroup() %>%
  unnest(hpi_hc) %>%
  filter(nchar(hpi_hc) > 100) %>%
  write_csv(file.path(path,'to_python','tbl_allnotes_chunked.csv.gz'))
  