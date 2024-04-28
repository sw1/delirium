pacman::p_load(tidyverse,textstem,tidytext,lubridate,icd.data)

# script to process all_icd_codes.csv file to create a table of icd codes

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

# load icd tables from icd.data package
icd9_lookup <- tibble(icd9cm_hierarchy) %>% mutate_all(str_to_lower)
icd10_lookup <- tibble(icd10cm2016) %>%  mutate_all(str_to_lower)

# create a list of words to search for in the long_desc field
words_lookup <- c('\\smental','consciou','encephalop','anxi','mood','halluc',
                  'psych','delus','deliri','dement','intox','depres',
                  'disorient')

# find icd9 codes that match the words in the long_desc field
icd9_match <- icd9_lookup[
  sapply(str_match_all(icd9_lookup$long_desc,paste(words_lookup,collapse='|')),
         length) > 0,
  ]

# filter specific codes that are not relevant
icd9_match <- icd9_match %>% 
  filter(!(three_digit %in% c('296','297','298','300','301','302','306','307',
                              '308','309','310','311','313','316','327','315',
                              '760',as.character(seq(800,899,1)),'995','v11',
                              'v15','v17','v62','v66','v67','v70','v79')))

# repeat for icd10
icd10_match <- icd10_lookup[
  sapply(str_match_all(icd10_lookup$long_desc,paste(words_lookup,collapse='|')),
         length) > 0,
  ]
icd10_match <- icd10_match %>% 
  filter(!(three_digit %in% c('a05','f06','f09','f22','f23','f24','f25','f28',
                              'f29','f30','f31','f32','f33','f34','f39','f40',
                              'f41','f43','f45','f48','f51','f53','f88','f89',
                              'j10','j11','o90','09a','f68','f93','o9a','s06',
                              't74','t76','z04','z56','z62','z64','z65','z69',
                              'z76','z81','z86','z91')))

# process the data from the icds.csv.gz file
# cleanup dates, times, and case identifiers
# match case icds with filtered icd lists then group them as lists
master <- read_rds(file.path(path,'data_in','00_icd_master.rds')) %>%
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


# save the icd table
write_rds(list(icds=master,
               icd9_match=icd9_match,icd10_match=icd10_match,
               words_lookup=words_lookup),
          file.path(path,'data_in','01_icd_tbl.rds'))