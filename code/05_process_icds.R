pacman::p_load(tidyverse,icd.data)

# script to merge icds and notes 

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

# load processed notes and labels, merge with icds then pivot to collapse
# icd9 and icd10 into single column. Keeping ids with only
# NAs because these are simply ids with no positive icd code
master <- read_csv(
  file.path(path,'to_python','tbl_to_python_expertupdate.csv.gz')) %>%
  select(id,label,set) %>%
  left_join(read_csv(file.path(path,'data_in','icds.csv.gz')) %>% 
              select(id=rdr_id,
                     icd9=diag_cd9,
                     icd10=diag_cd10),
            by='id') %>%
  mutate(across(contains('icd'),~str_replace_all(.x,'\\.| ','')),
         across(contains('icd'),~str_to_lower(.x))) %>% 
  pivot_longer(starts_with('icd'),names_to='code_type',values_to='code',
               values_drop_na=FALSE) %>%
  distinct() 

# filters icds in reference based on icds present in dataset
icd9_lookup <- tibble(icd9cm_hierarchy) %>% 
  mutate_all(str_to_lower) %>%
  filter(code %in% (master %>% 
                      filter(code_type == 'icd9') %>% 
                      distinct() %>%
                      pull(code)))
icd10_lookup <- tibble(icd10cm2016) %>%  
  mutate_all(str_to_lower) %>%
  filter(code %in% (master %>% 
                      filter(code_type == 'icd10') %>% 
                      distinct() %>%
                      pull(code)))

icds <- icd9_lookup %>% mutate(code_type='icd9') %>% 
  bind_rows(icd10_lookup %>% mutate(code_type='icd10')) %>% 
  select(code,code_type) %>%
  distinct()

# merge samples and their icds with their respective icd reference data
# then collapse icds into lists, then join notes
master <- master %>% 
  left_join(icds,by=c('code','code_type')) %>%
  group_by(id,set) %>%
  reframe(icd_codes=list(na.exclude(unique(code)))) %>%
  left_join(read_rds(file.path(path,'data_in','03_tbl_final_wperiods.rds')) %>%
              rename(icd_codes_del=icd_codes), 
            by='id') %>%
  distinct()

# final preprocessing before saving master table
master <- master %>%
  mutate(los=as.numeric(los),
         discharge_date=as.numeric(discharge_date),
         len_pmhx=as.numeric(len_pmhx),
         label=as.factor(label)) 


write_rds(master,file.path(path,'data_in','05_full_icd_tbl.rds'))


