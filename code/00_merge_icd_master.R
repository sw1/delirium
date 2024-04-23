library(tidyverse)

# script to merge icd and master table to obtain final ids for filtering
# based on date overlap according to the following rules:
# keep if Admission_date_hospital (notes file) <= 
# diagnosis_adm_dt (icd file) & 
# diagnosis_adm_dt (icd file) <= 
# Discharge_date_hospital (notes file)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

dir.create(path,showWarnings = FALSE, recursive = TRUE)

master <- read_csv(file.path(path,'data_in','notes.csv.gz')) %>%
  select(id=rdr_id,
         date_adm_master=Admission_date_hospital,
         date_dc_master=Discharge_date_hospital) %>%
  distinct() %>%
  mutate(date_adm_master=dmy(date_adm_master),
         date_dc_master=dmy(date_dc_master)) %>%
  left_join(read_csv(file.path(path,'data_in','icds.csv.gz')) %>%
              select(id=rdr_id,
                     mrn,
                     date_surg=Dateofsurgery,
                     date_adm_icd=diagnosis_adm_dt,
                     date_dc_icd=diagnosis_disch_dt) %>%
              distinct() %>%
              mutate(date_surg=mdy(date_surg),
                     date_adm_icd=mdy(date_adm_icd),
                     date_dc_icd=mdy(date_dc_icd)),
            by='id') %>%
  filter(date_adm_master <= date_adm_icd & date_adm_icd <= date_dc_master) %>%
  distinct() 

write_rds(master,file.path(path,'data_in','00_icd_master.rds'))
  