create_counts <- function(x,mem_save=FALSE,nurse=FALSE){
  
  cat('\nCreating counts.\n')
  x <- x %>%
    mutate(los=replace(los,is.infinite(los),NA),
           los=replace(los,is.na(los),mean(los,na.rm=TRUE))) %>%
    mutate(
      count_del=1*str_count(hospital_course,'deliri| cam |cows'),
      count_postdel=1*str_count(
        hospital_course,'postop deli|postoperative deli'),
      count_del_dd=1*str_count(discharge_diagnosis,'deliri'),
      count_del_ms=1*str_count(mental_status,'deliri'),
      count_del_prob=1*str_count(
        note,
        glue('(\\#\\s?deliri|\\d[:punct:]?\\s+delir)|',
             '(\\#\\s?postoperative deliri|',
             '\\d[:punct:]?\\s+postoperative delir)|(\\#\\s?postop deliri|',
             '\\d[:punct:]?\\s+postop delir)')),
      count_enceph_prob=1*str_count(
        note,
        '(\\#\\s?enceph|\\d[:punct:]?\\s+enceph)'),
      count_metenceph_prob=1*str_count(
        note,
        glue('(\\#\\s?met enceph|\\d[:punct:]?\\s+met enceph)|',
             '(\\#\\s?metabolic enceph|\\d[:punct:]?\\s+metabolic enceph)')),
      count_hepenceph_prob=1*str_count(
        note,
        glue('(\\#\\s?hep enceph|\\d[:punct:]?\\s+hep enceph)|',
             '(\\#\\s?hepatic enceph|\\d[:punct:]?\\s+hepatic enceph)')),
      count_toxenceph_prob=1*str_count(
        note,
        glue('(\\#\\s?tox enceph|\\d[:punct:]?\\s+tox enceph)|',
             '(\\#\\s?toxic enceph|\\d[:punct:]?\\s+toxic enceph)')),
      count_conf_hc=1*str_count(
        hospital_course,
        'confus|disorient|waxing|sundowni|sun downi|restrain|halluc'),
      count_conf_ms=1*str_count(
        mental_status,'confus|disorient|alter'),
      count_ao0_ms=1*str_count(
        mental_status,
        glue('((ao|oriented)\\s?x?\\s?(0|zero))|',
             '((ao|oriented)\\s?x?\\s?(1|one))|',
             '((ao|oriented)\\s?x?\\s?(2|two))')),
      count_ao3_ms=1*str_count(
        mental_status,'(ao|oriented)\\s?x?\\s?(3|three)'),
      count_ao0_hc=1*str_count(
        hospital_course,
        glue('((ao|oriented)\\s?x?\\s?(0|zero))|',
             '((ao|oriented)\\s?x?\\s?(1|one))|',
             '((ao|oriented)\\s?x?\\s?(2|two))')),
      count_ao3_hc=1*str_count(
        hospital_course,'(ao|oriented)\\s?x?\\s?(3|three)'),
      count_exf=1*str_count(discharge_disposition,'extend|servic'),
      count_hosp=1*str_count(discharge_disposition,'hospice|expir'),
      count_home=1*str_count(discharge_disposition,'home'),
      count_homeless=1*str_count(hospital_course,'homeless|shelter'),
      count_schiz=1*str_count(
        hospital_course,'schizo|delusion|disorganiz|cataton'),
      count_park=1*str_count(
        hospital_course,'parkins|dopa|duopa|rytary|sinemet'),
      count_alz=1*str_count(
        hospital_course,
        'alzh|brexpip|donepe|galant|memant|rivastig|aricept|exelon|razadyne'),
      count_manic=1*str_count(
        hospital_course,
        glue('manic|mania|bipol|lithium|lumateper|caplyta|idone|latuda|',
             'depakote|abilify|saphris|lamictal|aripipr|lamotrig')),
      count_enceph=1*str_count(hospital_course,'enceph'),
      count_nsurg=1*str_count(hospital_course,'neurosurg|craniot'),
      count_psych=1*str_count(hospital_course,'psychiatr'),
      count_geri=1*str_count(hospital_course,'geriatr'),
      count_pal=1*str_count(hospital_course,'paliat'),
      count_inf=1*str_count(
        hospital_course,'antibiot|bacteremi|mssa|mrsa|sepsi'),
      count_psych_med=1*str_count(
        hospital_course,
        glue('haloperidol|haldol|olanz|symbyax|precedex|dexmedet|',
             'seroquel|quetiapine')),
      count_ciwa=1*str_count(
        hospital_course,'ciwa|alcoho|withdraw|overdos|detox|tremens'),
      count_hep=1*str_count(
        hospital_course,
        'hepatit|hepatol|ascit|jaund|cirrh|varices|meld|portal'),
      count_tox=1*str_count(hospital_course,'toxic'),
      count_los=los                          
    ) 
  
  if (nurse){
    if (!mem_save){
      cat('Creating nursing counts.\n')
      x <- x %>% left_join(read_csv(
        file.path(path,'data_in','nurs_summaries.csv.gz')) %>% 
                             select(date_nurs=note_dt,mrn,
                                    note_nurs=note_txt) %>%
                             mutate(mrn=as.numeric(mrn)),
                           by='mrn',relationship='many-to-many') %>%
        mutate(date_nurs=ymd(date_nurs)) %>%
        rowwise() %>%
        mutate(note_overlap=if_else(
          date_nurs >= date_adm & date_nurs <= date_dc,1,0),
               note_nurs=if_else(note_overlap == 1,note_nurs,NA)) %>%
        ungroup() %>%
        group_by(id) %>%
        mutate(note_nurs=paste(note_nurs,collapse=' '),
               n_nurs=sum(note_overlap,na.rm=TRUE)) %>%
        ungroup() %>%
        select(-note_overlap,-date_nurs) %>%
        distinct() %>%
        mutate(
          count_nurse_del=1*str_count(
            note_nurs,'deliri| cam |cows'),
          count_nurse_conf_ms=1*str_count(
            note_nurs,
            'confus|disorient|waxing|sundowni|sun downi|restrain|halluc'),
          count_nurse_ao0_ms=1*str_count(
            note_nurs,
            glue('((ao|oriented)\\s?x?\\s?(0|zero))|',
                 '((ao|oriented)\\s?x?\\s?(1|one))|',
                 '((ao|oriented)\\s?x?\\s?(2|two))')),
          count_nurse_ao3_ms=1*str_count(
            note_nurs,
            '(ao|oriented)\\s?x?\\s?(3|three)'),
          count_nurse_pysch_med=1*str_count(
            note_nurs,
            glue('haloperidol|haldol|olanz|symbyax|precedex|dexmedet|',
                 'seroquel|quetiapine')),
          count_nurse_bp_med=1*str_count(
            note_nurs,
            glue('lithium|lumateper|caplyta|idone|latuda|depakote|abilify|',
                 'saphris|lamictal|aripipr|lamotrig')),
          count_nurse_alz_med=1*str_count(
            note_nurs,
            'brexpip|donepe|galant|memant|rivastig|aricept|exelon|razadyne'),
          count_nurse_wd=1*str_count(
            note_nurs,
            'ciwa|alcoho|withdraw|overdos|detox|tremens'),
          count_nurse_jaund=1*str_count(
            note_nurs,
            'ascit|jaund|cirrh|varices|meld')
        ) 
    }else{
      cat('Performing memory save.\n')
      cat('Saving tmp table.\n')
      write_rds(x,file.path(path,'data_tmp','raw_counts.rds'))
      
      x <- x %>% select(id,mrn,date_adm,date_dc) 
      
      cat('Clearing memory.\n')
      gc()
      
      cat('Creating nursing counts.\n')
      x <- x %>% 
        distinct() %>% 
        left_join(read_csv(
          file.path('D:\\Dropbox\\embeddings\\delirium',
                    'data_in',
                    'nurs_summaries.csv.gz')) %>% 
                    select(date_nurs=note_dt,mrn,note_nurs=note_txt) %>%
                    mutate(mrn=as.numeric(mrn)),
                  by='mrn',relationship='many-to-many') %>%
        mutate(date_nurs=ymd(date_nurs)) %>%
        rowwise() %>%
        mutate(note_overlap=if_else(
          date_nurs >= date_adm && date_nurs <= date_dc,1,0),
               note_nurs=if_else(note_overlap == 1,note_nurs,NA)) %>%
        ungroup() %>%
        group_by(id) %>%
        mutate(note_nurs=paste(note_nurs,collapse=' '),
               n_nurs=sum(note_overlap,na.rm=TRUE)) %>%
        ungroup() %>%
        select(-note_overlap,-date_nurs) %>%
        distinct() %>%
        mutate(
          count_nurse_del=1*str_count(
            note_nurs,
            'deliri| cam |cows'),
          count_nurse_conf_ms=1*str_count(
            note_nurs,
            'confus|disorient|waxing|sundowni|sun downi|restrain|halluc'),
          count_nurse_ao0_ms=1*str_count(
            note_nurs,
            glue('((ao|oriented)\\s?x?\\s?(0|zero))|',
                 '((ao|oriented)\\s?x?\\s?(1|one))|',
                 '((ao|oriented)\\s?x?\\s?(2|two))')),
          count_nurse_ao3_ms=1*str_count(
            note_nurs,
            '(ao|oriented)\\s?x?\\s?(3|three)'),
          count_nurse_pysch_med=1*str_count(
            note_nurs,
            glue('haloperidol|haldol|olanz|symbyax|precedex|dexmedet|',
                 'seroquel|quetiapine')),
          count_nurse_bp_med=1*str_count(
            note_nurs,
            glue('lithium|lumateper|caplyta|idone|latuda|depakote|abilify|',
                 'saphris|lamictal|aripipr|lamotrig')),
          count_nurse_alz_med=1*str_count(
            note_nurs,
            'brexpip|donepe|galant|memant|rivastig|aricept|exelon|razadyne'),
          count_nurse_wd=1*str_count(
            note_nurs,
            'ciwa|alcoho|withdraw|overdos|detox|tremens'),
          count_nurse_jaund=1*str_count(
            note_nurs,
            'ascit|jaund|cirrh|varices|meld')
        )
      
      cat('Merging tables.\n')
      x <- x %>%
        left_join(read_rds(file.path(path,'data_tmp','raw_counts.rds')),
                  by='id')
      
    }
  }
  
  return(x)
  
}

upsamp <- function(x,label_name='label'){
  
  x <- x %>%
    rename(label=label_name)
  
  n_1 <- sum(x$label == 1)
  n_0 <- sum(x$label == 0)
  n_upsamp_max <- max(c(n_0,n_1))
  n_upsamp_min <- min(c(n_0,n_1))
  n_upsamp <- round(n_upsamp_max/n_upsamp_min)
  train <- tibble()
  for (j in 1:n_upsamp){
    train <- train %>%
      bind_rows(x %>%
                  group_by(label) %>%
                  sample_n(n_upsamp_min,replace=TRUE) %>%
                  ungroup())
  }
  
  x <- x %>%
    rename(!!label_name := label)
  
  return(train)
  
}

get_rules <- function(x){
  y <- matrix('',nrow=nrow(x),ncol=3)
  for (i in 1:nrow(x)){
    d <- as.numeric(x[i,1])
    y[i,3] <- d
    if (d < 0.5){
      y[i,1] <- paste0('v',100-i)
    }else{
      y[i,1] <- paste0('v',nrow(x)-i+1)
    }
    y[i,2] <- gsub('\\s+',' ',
                   paste0(
                     gsub('is','==',x[i,3:ncol(x)]),collapse=' '))
    y[i,2] <- paste0('if_else(',y[i,2],',1,0)')
    
  }
  return(y)
}

chunk_nonoverlap <- function(x,n=4096){
  n_str <- nchar(x)
  n_ss <- ceiling(n_str/n)
  
  start <- 1
  end <- n
  out <- NULL
  while (length(out) <= n_ss){
    
    if (length(out) == (n_ss-1)) return(list(c(out,substr(x,start,n_str))))
    
    tmp_ss <- substr(x,start,end)
    ss_last_space <- tail(str_locate_all(tmp_ss,' ')[[1]],1)[1,1]
    end <- start + ss_last_space - 2
    
    out <- c(out,substr(x,start,end))
    start <- end + 2
    end <- start + n
  }
}

chunk <- function(x,n=4096){
  L <- nchar(x)
  R <- ceiling(L/n)
  skip <- floor(L/R)
  
  front <- str_trim(substr(x,1,n))
  back <- str_trim(substr(x,L-n+1,L))
  
  ss <- c(front,back)
  
  if (R > length(ss)){
    for (i in seq_len(R-2)){
      start <- i * skip
      end <- start + n - 1
      ss <- c(ss,str_trim(substr(x,start,end)))
    }
  }
  
  return(list(ss))
  
}



new_var <- function(df,v1,v2){
  df %>% mutate(!!rlang::parse_expr(v1) := !!rlang::parse_expr(v2))
}

update_df <- function(df,r){
  for (i in 1:nrow(r)){
    v1 <- r[i,1]
    v2 <- r[i,2]
    df <- new_var(df,v1,v2)
  }
  return(df)
}

get_icds <- function(x){
  z <- NULL
  for (i in 1:nrow(x)){
    r <- unlist(rules[i,])
    y <- r[startsWith(r,'icd_')]
    y <- gsub('icd_','',y)
    z <- c(z,y)
  }
  return(unique(z))
}

process_features <- function(x,n=50){
  feat_n <- nchar(x)
  y <- gsub(' |[[:punct:]]','',x)
  y <- paste0('icd_',y)
  y <- substr(y,1,n)
  y <- paste0(y,'_',feat_n)
  return(y)
}

z <- function(x) (x-mean(x,na.rm=TRUE))/sd(x,na.rm=TRUE)

med_list <- function(x){
  if (is.na(x)) return(list(NA))
  
  sep_comma <- str_count(x,'[:alnum:]\\, ')
  sep_digit <- str_count(x,'[:digit:]\\. ')
  
  if (sep_digit > sep_comma){
    x <- str_replace(x,'^.*?(?=\\d{1,2}\\.)','')
    x <- str_split(x,'\\d{1,2}\\.')[[1]]
    x <- str_replace_all(x,'[[:digit:]]|[[:punct:]]|\\$|\\^|\\+|\\<|\\>|\\=','')
    x <- str_squish(x)
    x <- str_match(x,'^([A-Za-z]+)*')[,2]
  }else if (sep_comma > sep_digit){
    x <- str_replace(x,'^.*?(?=\\s\\w+,)','')
    x <- str_split(x,',')[[1]]
    x <- str_replace_all(x,'[[:digit:]]|[[:punct:]]|\\$|\\^|\\+|\\<|\\>|\\=','')
    x <- str_squish(x)
    x <- str_match(x,'^([A-Za-z]+)*')[,2]
  }else if (sep_comma + sep_digit == 0){
    x <- str_replace_all(x,'[[:digit:]]|[[:punct:]]|\\$|\\^|\\+|\\<|\\>|\\=','')
    x <- str_squish(x)
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
  x <- str_squish(
    str_replace_all(x,'[[:punct:]][[:digit:]]|[[:digit:]]|\\*|\\:',''))
  x <- str_replace_all(x,' , ',', ')
  return(x)
}

allergy_list <- function(x){
  x <- str_squish(str_replace_all(x,'\\s+',' '))
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
  x <- str_squish(x)
}

icd_check <- function(code,reference){
  if (!is.na(code)){
    if (code %in% reference){
      return(code)
    } else {
      return(NA)
    }
  } else {
    return(NA)
  }
}

majority_vote <- function(predictions,labels){
  
  x <- tibble(id=rownames(predictions),pred=predictions[,1],lab=labels) %>%
    group_by(id) %>%
    mutate(max_pred=max(abs(0.5-pred))) %>%
    ungroup() %>%
    filter(abs(0.5-pred) == max_pred) %>%
    select(-max_pred) %>%
    mutate(pred=as.character(if_else(pred >= 0.5,1,0)))

  return(x)
}
