library(tidymodels)
library(glmnet)
library(tidyverse)
library(doParallel)
library(vip)
library(rpart.plot)
library(icd.data)

path <- 'D:\\Dropbox\\embeddings\\delirium'
ss <- 'icd'

# take table with all icd codes and case/id/mrn keys
# filter icd that do not correspond to the admission with surgery
# omits redundant notes based on dates
icd_full <- read_csv(file.path(path,'data_in','all_icd_codes.csv')) %>%
  mutate_at(7:12,~str_replace_all(.x,'\\.| ','')) %>%
  select(id=note_id,case=internalcaseid_deid_rdr,mrn=mrn_n,
         contains('diag'),
         date_surg=dateofsurgery,date_adm=adm_dt,date_dc=disch_dt) %>%
  mutate(date_surg=mdy(date_surg),
         date_adm=mdy(date_adm),
         date_dc=mdy(date_dc),
         current_admit=ifelse(date_surg >= date_adm & date_surg <= date_dc,1,0)) %>%
  filter(current_admit==1) %>%
  select(-current_admit) %>%
  mutate(los=date_dc-date_adm) %>%
  gather(key=code_type,value=code,-(id:mrn),-(date_surg:date_dc),-los,na.rm=TRUE) %>%
  distinct() %>%
  mutate(code_type=paste0('icd',str_extract(code_type,'\\d+$')),
         code=str_to_lower(code))

# filters icds in reference based on icds present in dataset
icd9_lookup <- tibble(icd9cm_hierarchy) %>% 
  mutate_all(str_to_lower) %>%
  filter(code %in% (icd_full %>% filter(code_type == 'icd9') %>% select(code) %>% distinct() %>% unlist()))
icd10_lookup <- tibble(icd10cm2016) %>%  
  mutate_all(str_to_lower) %>%
  filter(code %in% (icd_full %>% filter(code_type == 'icd10') %>% select(code) %>% distinct() %>% unlist()))


label_id <- read_csv(file.path(path,'to_python','tbl_to_python_updated.csv.gz')) %>%
  left_join(read_rds(file.path(path,'data_in','full_icd_tbl_allicds.rds')) %>% 
              select(id,note),
            by='id')

# rename icd subgroups that belong to one of the 3 bins defined above
icd_tbl <-  icd9_lookup %>% mutate(code_type='icd9') %>% 
  bind_rows(icd10_lookup %>% mutate(code_type='icd10')) %>% 
  select(code,code_type) %>%
  distinct()

# merge tables
# collapse icds into groups
tbl_full <- icd_full %>% 
  rename(note_id=id) %>% 
  left_join(read_csv(file.path(path,'data_in','Discharge Summaries ID+ ICD.csv')) %>%
              filter(note_type == 'DISCHARGE SUMMARY') %>%
              select(note_id=ïnote_id,case=internalcaseid_deid_rdr,
                     mrn=mrn_n,note=note_txt),
            by=c('note_id','mrn','case')) %>%
  left_join(icd_tbl,by=c('code','code_type')) %>%
  group_by(note) %>%
  summarize(date_adm=min(date_adm,na.rm=TRUE),
            date_dc=max(date_dc,na.rm=TRUE),
            los=date_dc-date_adm,
            icd_sub_chapter=list(na.omit(unique(code))))

# merge tables for remaining metadata
tbl_full <- tbl_full %>%
  inner_join(label_id,by='note') %>%
  left_join(read_rds(file.path(path,'data_in','tbl_final_wperiods.rds')) %>%
              select(id,mrn,n_note,icd_codes,activity_status,admission_date,
                     allergies,chief_complaint,date_of_birth,
                     discharge_condition,
                     discharge_date,discharge_diagnosis,discharge_disposition,
                     discharge_medications,family_history,
                     history_of_present_illness,
                     hospital_course,imaging,level_of_consciousness,
                     major_surgical_or_invasive_procedure,
                     medications_on_admission,
                     mental_status,past_medical_history,pertinent_results,
                     physical_exam,service,sex,social_history,
                     length_of_stay,age,num_meds,num_allergies,
                     len_pmhx,term_count_hc,term_count_hpi),
            by='id')


write_rds(tbl_full,file.path(path,'data_in',
                             sprintf('full_icd_tbl_%s.rds',gsub('_','',ss))))


