library(tidyverse)
library(stm)
library(tm)
library(textstem)
library(Rtsne)
library(parallel)
library(tidytext)
library(lubridate)
library(text2vec)
library(stopwords)
library(glmnet)
library(doParallel)
library(caret)

path <- 'C:\\Users\\sw424\\Shared\\desktop_research\\clean_workflow'

icd <- readRDS(file.path(path,'icd_tbl.rds'))$icd
conc <- read_csv(file.path(path,'concordance_labels.csv')) %>%
  select(case=internalcaseid_deid_rdr,label=adjudicator)

# tbl1 <- read_csv('C:/Users/sw424/Downloads/Data/NursingSummariesWithMRN_ID.csv')
tbl_full <- read_csv('C:/Users/sw424/Downloads/Data/Discharge Summaries ID+ ICD.csv') %>%
  filter(note_type == 'DISCHARGE SUMMARY') %>%
  select(id=ïnote_id,case=internalcaseid_deid_rdr,
         mrn=mrn_n,note=note_txt,icd=delirium_ICD) %>%
  left_join(icd,by=c('id','mrn','case')) %>%
  left_join(conc,by=c('case')) %>%
  distinct() %>%
  group_by(note,mrn) %>%
  filter(var(label) == 0 | is.na(var(label))) %>%
  summarize(icd_original=sum(icd,na.rm=TRUE),
            date_adm=min(date_adm,na.rm=TRUE),
            date_dc=max(date_dc,na.rm=TRUE),
            n_note=n(),
            label=unique(label),
            los=date_dc-date_adm,
            icd_sum=sum(icd_sum,na.rm=TRUE),
            icd_codes=list(unlist(list(icd_codes)))) %>%
  ungroup() %>%
  mutate(id=row_number())


saveRDS(tbl_full,file.path(path,'full_icd_tbl.rds'))





