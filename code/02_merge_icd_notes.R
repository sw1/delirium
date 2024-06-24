pacman::p_load(tidyverse,tm,textstem,tidytext,
               lubridate,text2vec,stopwords)

# read icd_table.rds containing all case ids and icd codes and merge
# that file with expert labels from notes.csv.gz and
# original haobo labels (concordance). 
# Since expert labels are only positive, will use the original 
# haobo negatives as negative examples (although will let any new label
# trump haobo labels). 

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

# the remaining notes that had duplicate labels in haobo table and did not
# have a corresponding expert label, so were manually annotated.
label_update_1 <- c()
label_update_0 <- c(617387)

# icd table.rds links to haobo labels via case ids whereas dc summaries and
# expert labels via note ids. Maintaining mrns to potentially
# link nursing table later. 

tbl <- read_csv(file.path(path,'data_in','notes.csv.gz')) %>%
  mutate(label=if_else(keyword_yn == 1,postop_delirium_yn,NA)) %>%
  select(id=rdr_id,label,note=note_txt,case=internalcaseid_deid_rdr) %>%
  left_join(read_rds(file.path(path,'data_out','01_icd_tbl.rds'))$icds,
            by='id') %>%
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
  mutate(label_icd=if_else(!is.na(icd_sum),
                           if_else(icd_sum > 0,1,0),
                           NA))
  
write_rds(tbl,file.path(path,'data_out','02_merged_icd_tbl.rds'))





