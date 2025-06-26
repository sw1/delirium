pacman::p_load(tidyverse,textstem,tidytext,lubridate,icd.data)

# script to process all_icd_codes.csv file to create a table of icd codes

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

# load icd tables from icd.data package
icd9_lookup <- tibble(icd9cm_hierarchy) %>% mutate_all(str_to_lower)
icd10_lookup <- tibble(icd10cm2016) %>%  mutate_all(str_to_lower)

#DOI: 10.1213/ANE.0000000000006425
icd9_del <- c('2809', '2900', '29011', '2903', '29041', 
              '2910', '2930', '2931', '29389')
icd10_del <- c('r410', 'f0390', 'f0391')

# filter specific codes that are not relevant
icd9_match <- icd9_lookup %>% 
  filter(code %in% icd9_del)

icd10_match <- icd10_lookup %>% 
  filter(code %in% icd10_del)

# process the data from the icds.csv.gz file
# cleanup dates, times, and case identifiers
# match case icds with filtered icd lists then group them as lists
master <- read_rds(file.path(path,'data_out','00_icd_master.rds')) %>%
  left_join(read_csv(file.path(path,'data_in','icds.csv.gz')) %>%
              select(id=rdr_id,
                     mrn,
                     date_surg=Dateofsurgery,
                     date_adm_icd=diagnosis_adm_dt,
                     date_dc_icd=diagnosis_disch_dt,
                     icd9=diag_cd9,
                     icd10=diag_cd10) %>%
              mutate(date_surg=mdy(date_surg),
                     date_adm_icd=mdy(date_adm_icd),
                     date_dc_icd=mdy(date_dc_icd)),
            by=c('id','mrn','date_surg','date_adm_icd','date_dc_icd')) %>%
  mutate(across(contains('icd'),~str_replace_all(.x,'\\.| ','')),
         across(contains('icd'),~str_to_lower(.x)),
         los=date_dc_master-date_adm_master) %>%
  filter(date_surg >= date_adm_master & date_surg <= date_dc_master) %>%
  rename(date_adm=date_adm_master,date_dc=date_dc_master) %>%
  select(-date_adm_icd,-date_dc_icd,-date_surg) %>%
  rowwise() %>%
  mutate(icd9=icd_check(icd9,icd9_match$code),
         icd10=icd_check(icd10,icd10_match$code)) %>%
  ungroup() %>%
  distinct() %>%
  group_by(id,mrn,date_adm,date_dc,los) %>%
  reframe(icd_codes=list(c(na.exclude(unique(c(icd9,icd10))))),
          icd_sum=lengths(icd_codes))


label_update_1 <- c()
label_update_0 <- c(617387)

tbl <- read_csv(file.path(path,'data_in','notes.csv.gz')) %>%
  mutate(label=if_else(keyword_yn == 1,postop_delirium_yn,NA)) %>%
  select(id=rdr_id,label,note=note_txt,case=internalcaseid_deid_rdr) %>%
  left_join(master,by='id') %>%
  left_join(read_csv(file.path(path,'data_in','concordance_labels.csv.gz')) %>%
              select(case=internalcaseid_deid_rdr,label_conc=adjudicator),
            by='case') %>%
  select(-case) %>%
  mutate(label=if_else(!is.na(label),label,
                       if_else(!is.na(label_conc),label_conc,NA))) %>%
  mutate(label=if_else(id %in% label_update_1,1,
                       if_else(id %in% label_update_0,0,
                               label))) %>%
  select(-label_conc) %>%
  distinct()

dups <- tbl %>% 
  group_by(note) %>% 
  filter(n()>1) 

tbl <- tbl %>% 
  anti_join(dups,by='id')

dups <- dups %>%
  group_by(note) %>%
  mutate(label=if (1 %in% label) {1} else {if (0 %in% label) {0} else {NA}},
         date_adm=min(date_adm),
         date_dc=max(date_dc),
         los=date_dc-date_adm,
         icd_codes=list(unique(unlist(icd_codes))),
         icd_sum=lengths(icd_codes),
         id=min(id)) %>% 
  distinct()

tbl <- bind_rows(tbl,dups) %>%
  mutate(label_icd=if_else(icd_sum > 0,1,0))

tbl %>% 
  select(id,label_pub_icd=label_icd) %>%
  write_csv(file.path(path,'data_out','02a_icd_tbl_pub.csv'))
