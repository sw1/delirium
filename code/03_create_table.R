library(tidyverse)
library(textstem)
library(Rtsne)
library(parallel)
library(tidytext)
library(lubridate)
library(tm)

path <- 'C:\\Users\\sw424\\Shared\\desktop_research\\clean_workflow'

med_list <- function(x){
  if (is.na(x)) return(list(NA))

  sep_comma <- str_count(x,'[:alnum:]\\, ')
  sep_digit <- str_count(x,'[:digit:]\\. ')

  if (sep_digit > sep_comma){
    x <- str_replace(x,'^.*?(?=\\d{1,2}\\.)','')
    x <- str_split(x,'\\d{1,2}\\.')[[1]]
    x <- str_replace_all(x,'[[:digit:]]|[[:punct:]]|\\$|\\^|\\+|\\<|\\>|\\=','')
    x <- trimws(x,'both')
    x <- str_match(x,'^([A-Za-z]+)*')[,2]
  }else if (sep_comma > sep_digit){
    x <- str_replace(x,'^.*?(?=\\s\\w+,)','')
    x <- str_split(x,',')[[1]]
    x <- str_replace_all(x,'[[:digit:]]|[[:punct:]]|\\$|\\^|\\+|\\<|\\>|\\=','')
    x <- trimws(x,'both')
    x <- str_match(x,'^([A-Za-z]+)*')[,2]
  }else if (sep_comma + sep_digit == 0){
    x <- str_replace_all(x,'[[:digit:]]|[[:punct:]]|\\$|\\^|\\+|\\<|\\>|\\=','')
    x <- trimws(x,'both')
    x <- str_match(x,'^([A-Za-z]+)*')[,2]
  }else{
    return(list(NA))
  }

  if (length(x) == 1 && (x == 'none' | is.na(x))) return(list(NA))

  x <- x[!is.na(x)]
  x <- x[nchar(x) > 2]
  x <- x[!(x %in% c('mg','mcg','g','ml'))]
  x <- list(x)
  return(x)
}

clean_text <- function(x){
  x <- stripWhitespace(trimws(str_replace_all(x,'[[:punct:]][[:digit:]]|[[:digit:]]|\\*|\\:',''),'both'))
  x <- str_replace_all(x,' , ',', ')
  return(x)
}

allergy_list <- function(x){
  x <- trimws(str_replace_all(x,'\\s+',' '))
  x <- unlist(str_split(x,' / '))
  len <- which(nchar(x) > 50)
  if (length(len) > 0) x <- x[len[1] - 1]
  x <- unique(x)
  x <- x[x != 'adverse drug reactions']
  x <- list(x)
  return(x)
}

aox <- function(x){
  x <- str_replace_all(x,'ao\\s?x?\\s?1|oriented\\s?x?\\s?1','aoxone')
  x <- str_replace_all(x,'ao\\s?x?\\s?2|oriented\\s?x?\\s?2','aoxtwo')
  x <- str_replace_all(x,'ao\\s?x?\\s?3|oriented\\s?x?\\s?3','aoxthree')
  x <- str_replace_all(x,'ao\\s?x?\\s?0|oriented\\s?x?\\s?0','aoxzero')
  x <- str_replace_all(x,'\\s+',' ')
  x <- str_replace_all(x,'[[:punct:]][[:punct::]]',' ')
  x <- str_replace_all(x,'[[:punct:]] [[:punct:]]',' ')
  x <- str_replace_all(x,' [[:punct:]]',' ')
  x <- str_replace_all(x,'\\s+',' ')
}

pmhx <- function(x){
  x <- str_replace_all(x,'\\/','')
  x <- str_replace_all(x,'[[:digit:]]|[[:punct:]]',' ')
  x <- str_replace_all(x,'\\s+',' ')
  x <- trimws(x,'both')
}


tbl_full <- readRDS(file.path(path,'full_icd_tbl.rds'))

breaks <- c(seq(1,nrow(tbl_full),by=20000),nrow(tbl_full))
for (i in 1:(length(breaks)-1)){
  cat(sprintf('Splitting notes for break %s.\n',as.character(i)))
  tbl <- tbl_full[breaks[i]:breaks[i+1],] %>%
    mutate(note=tolower(note)) %>%
    unnest_tokens(sentence,note,
                  token='regex',
                  pattern='(?=>unit no\\n|admission date\\:|discharge date\\:|date of birth\\:|sex\\:|service\\:|major surgical or invasive procedure\\:|allergies\\:|attending\\:|chief complaint\\:|history of present illness\\:|past medical history\\:|social history\\:|family history\\:|physical exam\\:|pertinent results\\:|imaging\\:|hospital course\\:|medications on admission\\:|discharge medications\\:|discharge disposition\\:|discharge diagnosis\\:|discharge condition\\:|mental status\\:|level of consciousness\\:|activity status\\:|discharge instructions\\:)',
                  to_lower=FALSE,drop=TRUE) %>%
    separate(sentence,c('section','text'),sep='\\:',remove=TRUE,extra='merge',fill='right') %>%
    filter(!grepl('^discharge summary.*|^discharge instructions.*|attending',section)) %>%
    mutate(section=str_replace_all(section,' ','_')) %>%
    group_by(id,section) %>%
    mutate(text=paste(text,collapse=' ')) %>%
    distinct() %>%
    spread(section,text)

  saveRDS(tbl,file.path(path,sprintf('tbl_tmp1_%s.rds',as.character(i))))

}

for (i in 1:(length(breaks)-1)){
  cat(sprintf('Parsing data for break %s.\n',as.character(i)))

  tbl <- readRDS(file.path(path,sprintf('tbl_tmp1_%s.rds',as.character(i)))) %>%
      rowwise() %>%
      mutate(medications_on_admission=med_list(medications_on_admission),
             allergies=allergy_list(allergies)) %>%
      ungroup() %>%
      mutate_if(is.character,~trimws(str_replace_all(.x,'\\$|\\^|\\+|\\<|\\>|\\=|\\%|\\#',''),'both')) %>%
      mutate(admission_date=mdy(admission_date),
             discharge_date=mdy(discharge_date),
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
             service=str_replace_all(trimws(service),'\\s+.*',''),
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

    saveRDS(tbl,file.path(path,sprintf('tbl_tmp2_%s.rds',as.character(i))))
}


for (i in 1:(length(breaks)-1)){
  cat(sprintf('Counting data for break %s.\n',as.character(i)))

  tbl <- readRDS(file.path(path,sprintf('tbl_tmp2_%s.rds',as.character(i))))  %>%
    mutate(term_count_hc=str_count(hospital_course,'deliri|combat|sedat|unconc|briefly awak|brief awak|not fully alert|pulling out|pulled out|aggress|disorgan|altered|fluctuat|confus|encephalopathic|sundowni|sun downi|disorient|reoritent|agitat|disinhib|aox(one|two|three)|haloperidol|haldol|olanz|precedex|dexmedetomidine|restrain|seroquel|quetiapine|constipat|oliguri|miralax|laxative|polyethylene|insomn|remelteon|melatonin|hearing aid|glasses|ativan|lorazepam|valium|diazepam|tranlat|pt|physical therap|occupational therap|wound care|wound nurs|palliativ|geriatri|psych|rass|cam|ciwa|cows|fall|fell|tranferr|rehab|ambian|trazadone'),
           term_count_hpi=str_count(history_of_present_illness,'deliri|unconc|sedat|briefly awak|brief awak|not fully alert|aggress|combat|disorgan|altered|fluctuat|confus|encephalopathic|sundowni|sun downi|disorient|reoritent|agitat|disinhib|aox(one|two|three)|haloperidol|haldol|olanz|precedex|dexmedetomidine|restrain|seroquel|quetiapine|constipat|oliguri|miralax|laxative|polyethylene|insomn|remelteon|melatonin|hearing aid|glasses|ativan|lorazepam|valium|diazepam|tranlat|pt|physical therap|occupational therap|wound care|wound nurs|palliativ|geriatri|psych|cam|ciwa|cows|fall|fell|tranferr|rehab|readmi|nursing home|nursing facility|snf|caretak|caregiv|polypharm|drug use|drug abuse|narcotic|ivdu|alcohol|subox|methadon|shelter|homeless|divorc|widow|withdrawal|disheveled|paranoi|scared|frightened|miosis|pinpoint'))

  saveRDS(tbl,file.path(path,sprintf('tbl_tmp3_%s.rds',as.character(i))))
}


tbl <- readRDS(file.path(path,'tbl_tmp3_1.rds'))
for (i in 2:(length(breaks)-1)){
  tbl <- tbl %>%
    bind_rows(readRDS(file.path(path,sprintf('tbl_tmp3_%s.rds',as.character(i)))))

}
tbl <- tbl %>%
  distinct() %>%
  rowwise() %>%
  mutate(hpi_hc=paste(c(history_of_present_illness,hospital_course),collapse=' ')) %>%
  mutate(icd_sum=ifelse(is.na(icd_sum),icd_original,icd_sum)) %>%
  ungroup() %>%
  filter(!is.na(history_of_present_illness), nchar(history_of_present_illness) >= 150) %>%
  filter(!is.na(hospital_course), nchar(hospital_course) >= 150)

saveRDS(tbl,file.path(path,'tbl_final_wperiods.rds'))

tbl <- tbl %>%
  mutate_at(c('history_of_present_illness','hospital_course'),
            ~stripWhitespace(trimws(str_replace_all(.x,'[[:punct:]]',''),'both')))

saveRDS(tbl,file.path(path,'tbl_final.rds'))
