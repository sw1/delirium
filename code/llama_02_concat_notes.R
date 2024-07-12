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
  group_by(id) %>%
  arrange(desc(note_date)) %>%
  reframe(note=paste(note,collapse=' ')) %>%
  left_join(tbl,by='id') %>%
  mutate(hpi_hc = glue('{hc} {note}'),
         len=nchar(hpi_hc),
         n_tokens=length(str_split(hpi_hc,' ')[[1]])) %>%
  filter(nchar(hpi_hc) > 100) %>% 
  filter(n_tokens <= 8192) %>%
  select(all_of(colnames(tbl))) %>%
  write_csv(file.path(path,'to_python','tbl_allnotes_trimmed.csv.gz'))

notes <- notes %>%
  group_by(id) %>%
  arrange(note_date) %>%
  reframe(note=paste(note,collapse=' ')) %>%
  left_join(tbl,by='id') %>%
  mutate(hpi_hc = glue('{hc} {note}'),
         len=nchar(hpi_hc),
         n_tokens=length(str_split(hpi_hc,' ')[[1]])) %>%
  filter(nchar(hpi_hc) > 100) %>%
  select(all_of(colnames(tbl))) %>%
  write_csv(file.path(path,'to_python','tbl_allnotes.csv.gz'))
