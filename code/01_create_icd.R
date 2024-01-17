library(tidyverse)
library(textstem)
library(Rtsne)
library(parallel)
library(tidytext)
library(lubridate)
library(icd.data)

path <- 'C:\\Users\\sw424\\Shared\\desktop_research\\clean_workflow'
dir.create(path,showWarnings = FALSE, recursive = TRUE)

icd9_lookup <- tibble(icd9cm_hierarchy) %>% mutate_all(str_to_lower)
icd10_lookup <- tibble(icd10cm2016) %>%  mutate_all(str_to_lower)

words_lookup <- c('\\smental','consciou','encephalop','anxi','mood','halluc','psych','delus','deliri','dement','intox','depres','disorient')

icd9_match <- icd9_lookup[sapply(str_match_all(icd9_lookup$long_desc,paste(words_lookup,collapse='|')),length) > 0,]
icd9_match <- icd9_match %>% filter(!(three_digit %in% c('296','297','298','300','301','302','306','307','308','309','310','311','313','316','327','315','760',as.character(seq(800,899,1)),'995','v11','v15','v17','v62','v66','v67','v70','v79')))

icd10_match <- icd10_lookup[sapply(str_match_all(icd10_lookup$long_desc,paste(words_lookup,collapse='|')),length) > 0,]
icd10_match <- icd10_match %>% filter(!(three_digit %in% c('a05','f06','f09','f22','f23','f24','f25','f28','f29','f30','f31','f32','f33','f34','f39','f40','f41','f43','f45','f48','f51','f53','f88','f89','j10','j11','o90','09a','f68','f93','o9a','s06','t74','t76','z04','z56','z62','z64','z65','z69','z76','z81','z86','z91')))


options(dplyr.summarise.inform=FALSE)
f <- function(x,pos){
  x %>%
    mutate_at(7:12,~str_replace_all(.x,'\\.| ','')) %>%
    select(id=note_id,case=internalcaseid_deid_rdr,mrn=mrn_n,
           icd9=diag_cd9,icd10=diag_cd10,
           date_surg=dateofsurgery,date_adm=adm_dt,date_dc=disch_dt) %>%
    mutate(date_surg=mdy(date_surg),
           date_adm=mdy(date_adm),
           date_dc=mdy(date_dc),
           current_admit=ifelse(date_surg >= date_adm & date_surg <= date_dc,1,0)) %>%
    filter(current_admit==1) %>%
    mutate(icd9=str_to_lower(icd9),
           icd10=str_to_lower(icd10),
           los=date_dc-date_adm) %>%
    rowwise() %>%
    mutate(code_diag=ifelse((icd9 %in% icd9_match$code) || (icd10 %in% icd10_match$code),1,0)) %>%


    ungroup() %>%
    mutate(icd9=ifelse(code_diag == 1,icd9,NA),
           icd10=ifelse(code_diag == 1,icd10,NA)) %>%


    group_by(case,id,mrn,date_surg,date_adm,date_dc,los) %>%
    summarize(icd_sum=sum(code_diag,na.rm=TRUE),
              icd_codes=ifelse(icd_sum==0,NA,list(c(na.exclude(c(unique(icd9),unique(icd10)))))))
}
icd <- read_csv_chunked('C:/Users/sw424/Downloads/Data/all_icd_codes.csv',
                        DataFrameCallback$new(f),col_names=TRUE,
                        chunk_size=250000,progress=show_progress())

saveRDS(list(icd=icd,icd9_match=icd9_match,icd10_match=icd10_match,
             words_lookup=words_lookup),
        file.path(path,'icd_tbl.rds'))