library(tidyverse)
library(textstem)
library(parallel)
library(tidytext)
library(lubridate)
library(tm)
library(glue)

# cript to preprocess notes into specific categories and remove various
# types of characters etc.

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

tbl_full <- read_rds(file.path(path,'data_in','full_icd_tbl.rds'))

# split notes into sections by iterating over 200000 rows at a time
# to save memory.
breaks <- c(seq(1,nrow(tbl_full),by=20000),nrow(tbl_full))
for (i in 1:(length(breaks)-1)){
  cat(glue('Splitting notes for break {i}.\n\n'))
  tbl <- tbl_full[breaks[i]:breaks[i+1],] %>%
    mutate(note=str_replace_all(note,'\\r',' '),
           note=tolower(note)) %>%
    unnest_tokens(sentence,note,
                  token='regex',
                  pattern=glue('(?=>unit no\\n|admission date\\:|',
                               'discharge date\\:|date of birth\\:|',
                               'sex\\:|service\\:|',
                               'major surgical or invasive procedure\\:|',
                               'allergies\\:|attending\\:|chief complaint\\:|',
                               'history of present illness\\:|',
                               'past medical history\\:|social history\\:|',
                               'family history\\:|physical exam\\:|',
                               'pertinent results\\:|imaging\\:|',
                               'hospital course\\:|',
                               'medications on admission\\:|',
                               'discharge medications\\:|',
                               'discharge disposition\\:|',
                               'discharge diagnosis\\:|',
                               'discharge condition\\:|mental status\\:|',
                               'level of consciousness\\:|activity status\\:|',
                               'discharge instructions\\:)'),
                  to_lower=FALSE,drop=TRUE) %>%
    separate(sentence,c('section','text'),sep='\\:',
             remove=TRUE,extra='merge',fill='right') %>%
    filter(!grepl(glue('^discharge summary.*|^discharge instructions.*|',
                       'attending|^addendum to discharge.*'),
                  section)) %>%
    mutate(section=str_replace_all(section,' ','_')) %>%
    group_by(id,section) %>%
    mutate(text=paste(text,collapse=' ')) %>%
    distinct() %>%
    spread(section,text)

  write_rds(tbl,file.path(path,'data_tmp',glue('tbl_tmp1_{i}.rds')))
}

# load tmp files and parse note sections to extract features and
# additional sections
for (i in 1:(length(breaks)-1)){
  cat(glue('Parsing data for break {i}.\n\n'))
  tbl <- read_rds(file.path(path,'data_tmp',glue('tbl_tmp1_{i}.rds'))) %>%
      rowwise() %>%
      mutate(medications_on_admission=med_list(medications_on_admission),
             allergies=allergy_list(allergies)) %>%
      ungroup() %>%
      mutate_if(is.character,
                ~str_squish(str_replace_all(
                  .x,'\\$|\\^|\\+|\\<|\\>|\\=|\\%|\\#',''))) %>%
      mutate(admission_date=mdy(admission_date),
             discharge_date=mdy(word(discharge_date)),
             length_of_stay=discharge_date-admission_date,
             date_of_birth=mdy(date_of_birth),
             sex=ifelse(sex=='m','male',ifelse(sex=='f','female','other')),
             age=(admission_date-date_of_birth)/365,
             num_meds=lengths(medications_on_admission),
             num_allergies=lengths(allergies),
             len_pmhx=nchar(past_medical_history),
             history_of_present_illness=aox(history_of_present_illness),
             hospital_course=aox(hospital_course),
             past_medical_history=pmhx(past_medical_history),
             service=str_replace_all(str_squish(service),'\\s+.*',''),
             service=case_when(
               service %in% c('','admission') ~ 'other',
               service == 'sur' ~ 'surgery',
               service == 'ort' ~ 'orthopaedics',
               service %in% c('gyn','ob') ~ 'obstetrics/gynecology',
               TRUE ~ service)) %>%
    mutate(history_of_present_illness=clean_text(history_of_present_illness),
           hospital_course=clean_text(hospital_course)) %>%
    group_by(service) %>%
    mutate(n_service=length(service)) %>%
    mutate(service=if_else(n_service < 100,'other',service)) %>%
    ungroup() %>%
    select(-n_service)

    write_rds(tbl,file.path(path,'data_tmp',glue('tbl_tmp2_{i}.rds')))
}

# load tmp files and begin term counts
for (i in 1:(length(breaks)-1)){
  cat(glue('Counting data for break {i}.\n\n'))
  tbl <- read_rds(file.path(path,'data_tmp',glue('tbl_tmp2_{i}.rds')))  %>%
    mutate(term_count_hc=str_count(hospital_course,
                                   glue('deliri|combat|sedat|unconc|',
                                        'briefly awak|brief awak|',
                                        'not fully alert|pulling out|',
                                        'pulled out|aggress|disorgan|altered|',
                                        'fluctuat|confus|encephalopathic|',
                                        'sundowni|sun downi|disorient|',
                                        'reoritent|agitat|disinhib|',
                                        'aox(one|two|three)|haloperidol|',
                                        'haldol|olanz|precedex|',
                                        'dexmedetomidine|restrain|seroquel|',
                                        'quetiapine|constipat|oliguri|',
                                        'miralax|laxative|polyethylene|',
                                        'insomn|remelteon|melatonin|',
                                        'hearing aid|glasses|ativan|',
                                        'lorazepam|valium|diazepam|tranlat|',
                                        'pt|physical therap|',
                                        'occupational therap|wound care|',
                                        'wound nurs|palliativ|geriatri|',
                                        'psych|rass|cam|ciwa|cows|fall|fell|',
                                        'tranferr|rehab|ambian|trazadone')),
           term_count_hpi=str_count(history_of_present_illness,
                                    glue('deliri|unconc|sedat|briefly awak|',
                                         'brief awak|not fully alert|aggress',
                                         '|combat|disorgan|altered|fluctuat|',
                                         'confus|encephalopathic|sundowni|',
                                         'sun downi|disorient|reoritent|',
                                         'agitat|disinhib|aox(one|two|three)|',
                                         'haloperidol|haldol|olanz|precedex|',
                                         'dexmedetomidine|restrain|seroquel|',
                                         'quetiapine|constipat|oliguri|',
                                         'miralax|laxative|polyethylene|',
                                         'insomn|remelteon|melatonin|',
                                         'hearing aid|glasses|ativan|',
                                         'lorazepam|valium|diazepam|tranlat|',
                                         'pt|physical therap|',
                                         'occupational therap|wound care|',
                                         'wound nurs|palliativ|geriatri|',
                                         'psych|cam|ciwa|cows|fall|fell|',
                                         'tranferr|rehab|readmi|nursing home|',
                                         'nursing facility|snf|caretak|',
                                         'caregiv|polypharm|drug use|',
                                         'drug abuse|narcotic|ivdu|alcohol|',
                                         'subox|methadon|shelter|homeless|',
                                         'divorc|widow|withdrawal|disheveled|',
                                         'paranoi|scared|frightened|miosis|',
                                         'pinpoint')))

  write_rds(tbl,file.path(path,'data_tmp',glue('tbl_tmp3_{i}.rds')))
}

# load tmp files and combine into final table
tbl <- read_rds(file.path(path,'data_tmp','tbl_tmp3_1.rds'))
for (i in 2:(length(breaks)-1)){
  tbl <- tbl %>%
    bind_rows(read_rds(file.path(path,'data_tmp',glue('tbl_tmp3_{i}.rds'))))

}
tbl <- tbl %>%
  distinct() %>%
  rowwise() %>%
  mutate(hpi_hc=paste(c(history_of_present_illness,hospital_course),
                      collapse=' ')) %>%
  ungroup() 

# save the final table with period
write_rds(tbl,file.path(path,'data_in','tbl_final_wperiods.rds'))

# remove periods from the final table and save
tbl <- tbl %>%
  mutate_at(c('history_of_present_illness','hospital_course'),
            ~str_squish(trimws(str_replace_all(.x,'[[:punct:]]',''))))

write_rds(tbl,file.path(path,'data_in','tbl_final.rds'))
