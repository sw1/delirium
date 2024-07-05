pacman::p_load(tidyverse,textstem,tidytext,lubridate,tm,glue, data.table)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

years <- list.files(file.path(path,'data_in'),pattern='notes_[0-9]+\\.csv.gz')
years <- unique(str_extract(years,'[0-9]+'))

tbl_include <- read_csv(file.path(path,'to_python','tbl.csv.gz')) %>%
  select(id) %>%
  left_join(read_rds(file.path(path,'data_out','03_tbl_final.rds')) %>%
              select(id,mrn,admission_date,discharge_date),by=c('id')) 

for (year in years){
  read_csv(file.path(path,'data_in',glue('notes_{year}.csv.gz'))) %>%
    select(mrn,note_date=note_dt,type=note_type_full,service,note=note_txt) %>%
    mutate(mrn=as.numeric(mrn),
           type=tolower(type),
           service=tolower(service),
           note_date=ymd(note_date)) %>%
    filter(!is.na(mrn),
           type %in% c('event','initial note','post operative',
                       'progress note','transfer','final/sign off'),
           !(service %in% c('neonatology',
                            'walk-in',
                            'clinical research center',
                            'admitting', 
                            'case management', #maybe?
                            'employee occ health',
                            'pathology',
                            'ethics',
                            'health care quality',
                            'genetic counseling',
                            'him- medical records',
                            'home care',
                            'infecious control',
                            'integrated care',
                            'office of business conduct',
                            'nursing', #not avail for early year
                            'pediatrics',
                            'periop services', 
                            'pherisis',
                            'prenatal genetics',
                            'rehabilitative services', #maybe?
                            'social work',
                            'spiritual care', #maybe?
                            'travel clinic',
                            'walk-in',
                            'wound/ostomy', 
                            'null',
                            'pharmacy',
                            'radiology',
                            'nutrition'))) %>%
    inner_join(tbl_include,by='mrn',relationship='many-to-many') %>%
    filter((note_date >= admission_date) & (note_date <= discharge_date)) %>%
    write_rds(file.path(path,'data_tmp',glue('notes_filt_{year}.rds')))
  
}
