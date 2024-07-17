pacman::p_load(tidyverse,textstem,tidytext,lubridate,tm,glue,doParallel)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))


tbl <- read_csv(file.path(path,'to_python','tbl.csv.gz')) 

years <- list.files(file.path(path,'data_tmp'),
                    pattern='notes_proc_[0-9]+\\.rds')
years <- unique(str_extract(years,'[0-9]+'))

notes <- tibble()
for (year in years){
  notes <- notes %>% bind_rows(read_rds(
    file.path(path,'data_tmp',glue('notes_proc_{year}.rds')))
  )
}

notes %>%
  distinct(note,.keep_all = TRUE) %>%
  group_by(id) %>%
  arrange(desc(note_date)) %>%
  reframe(note=paste(note,collapse=' ')) %>%
  left_join(tbl,by='id') %>%
  mutate(hpi_hc = glue('{hc} {note}'),
         len=nchar(hpi_hc),
         n_tokens=lengths(str_split(hpi_hc,' '))) %>%
  filter(nchar(hpi_hc) > 100) %>% 
  filter(n_tokens <= 8192) %>%
  select(all_of(colnames(tbl))) %>%
  mutate(hpi='',hc='') %>%
  write_csv(file.path(path,'to_python','tbl_allnotes_trimmedtoken.csv.gz'))

notes %>%
  distinct(note,.keep_all = TRUE) %>%
  arrange(id,note_date) %>%
  left_join(tbl,by='id') %>%
  select(note,all_of(colnames(tbl)),-hpi_hc) %>%
  group_by(id) %>% 
  pivot_longer(matches(c('^hpi$','^hc$','^note$')),
               names_to='type',values_to='note') %>%
  filter(nchar(note) > 100) %>%
  mutate(type=if_else(type == 'hpi',1,if_else(type == 'hc',3,2))) %>%
  arrange(id,type) %>%
  ungroup() %>%
  rename(hpi_hc=note) %>%
  mutate(hpi='',hc='') %>%
  select(all_of(colnames(tbl))) %>%
  distinct() %>%
  write_csv(file.path(path,'to_python','tbl_allnotes_uncat.csv.gz'))

notes <- notes %>%
  distinct(note,.keep_all = TRUE) %>%
  group_by(id) %>%
  arrange(note_date) %>%
  reframe(note=paste(note,collapse=' ')) %>%
  left_join(tbl,by='id') %>%
  mutate(hpi_hc = glue('{hc} {note}')) %>%
  filter(nchar(hpi_hc) > 100) %>%
  select(all_of(colnames(tbl))) %>%
  mutate(hpi='',hc='') 

notes %>%
  write_csv(file.path(path,'to_python','tbl_allnotes.csv.gz'))

notes %>%
  mutate(hpi_hc=substr(hpi_hc,1,120000)) %>%
  write_csv(file.path(path,'to_python','tbl_allnotes_trimmedlen.csv.gz'))
