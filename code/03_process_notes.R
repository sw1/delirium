pacman::p_load(tidyverse,textstem,tidytext,lubridate,tm,glue)

# script to preprocess notes into specific categories and remove various
# types of characters etc.

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

tbl_full <- read_rds(file.path(path,'data_out','02_merged_icd_tbl.rds'))

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
                                        'aox(zero|one)|haloperidol|',
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
                                         'agitat|disinhib|aox(zero|one)|',
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
  mutate(history_of_present_illness=if_else(
    is.na(history_of_present_illness),'',history_of_present_illness),
         hpi_hc=paste(c(history_of_present_illness,hospital_course),
                      collapse=' ')) %>%
  ungroup() 

# mutate some variables for st
tbl <- tbl %>%
  mutate(los=if_else(!is.na(length_of_stay),length_of_stay,los),
         sex=if_else(sex == 'female',0,if_else(sex == 'male',1,NA)),
         age=if_else(!is.na(age),age,date_of_birth-admission_date))

write_rds(tbl,file.path(path,'data_out','03_tbl_final_beforeinterp.rds'))

# interp age, sex, los, discharge/admission dates, term_count_hpi
tbl <- tbl %>%
  mutate(bucket_age=ntile(age,10),
         bucket_num_meds=ntile(num_meds,4),
         bucket_len_pmhx=ntile(len_pmhx,4)) %>%
  group_by(across(starts_with('bucket')),service) %>%
  mutate(los=if_else(is.na(los),mean(los,na.rm=TRUE),los)) %>%
  ungroup() %>%
  mutate(bucket_los=ntile(los,4)) %>%
  group_by(bucket_num_meds,bucket_len_pmhx,bucket_los,service) %>%
  mutate(age=if_else(is.na(age),mean(age,na.rm=TRUE),age)) %>%
  ungroup() %>%
  mutate(bucket_age=ntile(age,10)) %>%
  mutate(discharge_date=if_else(is.na(discharge_date),
                                date_dc,
                                discharge_date),
         admission_date=if_else(is.na(admission_date),
                                date_adm,
                                admission_date),
         discharge_date=if_else(is.na(discharge_date),
                                admission_date + los,
                                discharge_date),
         admission_date=if_else(is.na(admission_date),
                                discharge_date - los,
                                admission_date)) %>%
  select(-date_adm,-date_dc,length_of_stay) %>% 
  mutate(age=as.integer(round(age)),
         age=if_else(id == 477879,40,age),
         admission_date=if_else(id == 477879,
                                date_of_birth + age,
                                admission_date),
         discharge_date=if_else(id == 477879,
                                admission_date + los,
                                discharge_date)) %>%
  group_by(across(starts_with('bucket')),service) %>%
  mutate(term_count_hpi=if_else(is.na(term_count_hpi),
                                round(mean(term_count_hpi,na.rm=TRUE)),
                                term_count_hpi)) %>%
  ungroup() %>%
  select(-starts_with('bucket'))

# fix sex missing values
tbl <- tbl %>%
  mutate(sex=if_else(is.na(sex),
                     case_when(
                       service == 'obstetrics/gynecology' ~ 0,
                       str_detect(hpi_hc,glue('^ms|^mrs|^miss|^f |',
                                              '^sister |yo f')) ~ 0,
                       str_detect(hpi_hc,'^mr|^m |^father |yo m') ~ 1,
                       str_detect(hpi_hc,'female|woman|lady') ~ 0,
                       str_detect(hpi_hc,'male|gentleman|man') ~ 1,
                       str_detect(hpi_hc,'she|her') ~ 1,
                       str_detect(hpi_hc,'he|his') ~ 0,
                       TRUE ~ NA),
                     sex)) 

# estimate len_pmhx if missing
tbl <- tbl %>%
  mutate(len_pmhx=ifelse(is.na(len_pmhx),
                         estimate_len_pmhx(hpi_hc),
                         len_pmhx)) 

# filter hc <= 50 or concat hpi_hc <= 100 from full table
tbl <- tbl %>%
  filter(
    nchar(hospital_course) > 50 | str_detect(hospital_course,'see hpi')) %>%
  filter(nchar(hpi_hc) > 100)

# save the final table with period
write_rds(tbl,file.path(path,'data_out','03_tbl_final_wperiods.rds'))

# remove periods from the final table and save
tbl <- tbl %>%
  mutate_at(c('history_of_present_illness','hospital_course'),
            ~str_squish(trimws(str_replace_all(.x,'[[:punct:]]',''))))

write_rds(tbl,file.path(path,'data_out','03_tbl_final.rds'))

unlink(file.path(path,'data_tmp','*'),recursive=TRUE,force=TRUE)
