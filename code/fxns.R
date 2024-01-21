create_counts <- function(x,mem_save=FALSE){
  
  cat('\nCreating counts.\n')
  x <- x %>%
    mutate(los=replace(los,is.infinite(los),NA),
           los=replace(los,is.na(los),mean(los,na.rm=TRUE))) %>%
    mutate(
      count_del=1*str_count(hc,'deliri| cam |cows'),
      count_postdel=1*str_count(hc,'postop deli|postoperative deli'),
      count_del_dd=1*str_count(discharge_diagnosis,'deliri'),
      count_del_ms=1*str_count(mental_status,'deliri'),
      count_del_prob=1*str_count(note,'(\\#\\s?deliri|\\d[:punct:]?\\s+delir)|(\\#\\s?postoperative deliri|\\d[:punct:]?\\s+postoperative delir)|(\\#\\s?postop deliri|\\d[:punct:]?\\s+postop delir)'),
      count_enceph_prob=1*str_count(note,'(\\#\\s?enceph|\\d[:punct:]?\\s+enceph)'),
      count_metenceph_prob=1*str_count(note,'(\\#\\s?met enceph|\\d[:punct:]?\\s+met enceph)|(\\#\\s?metabolic enceph|\\d[:punct:]?\\s+metabolic enceph)'),
      count_hepenceph_prob=1*str_count(note,'(\\#\\s?hep enceph|\\d[:punct:]?\\s+hep enceph)|(\\#\\s?hepatic enceph|\\d[:punct:]?\\s+hepatic enceph)'),
      count_toxenceph_prob=1*str_count(note,'(\\#\\s?tox enceph|\\d[:punct:]?\\s+tox enceph)|(\\#\\s?toxic enceph|\\d[:punct:]?\\s+toxic enceph)'),
      count_conf_hc=1*str_count(hc,'confus|disorient|waxing|sundowni|sun downi|restrain|halluc'),
      count_conf_ms=1*str_count(mental_status,'confus|disorient|alter'),
      count_ao0_ms=1*str_count(mental_status,'((ao|oriented)\\s?x?\\s?(0|zero))|((ao|oriented)\\s?x?\\s?(1|one))|((ao|oriented)\\s?x?\\s?(2|two))'),
      count_ao3_ms=1*str_count(mental_status,'(ao|oriented)\\s?x?\\s?(3|three)'),
      count_ao0_hc=1*str_count(hc,'((ao|oriented)\\s?x?\\s?(0|zero))|((ao|oriented)\\s?x?\\s?(1|one))|((ao|oriented)\\s?x?\\s?(2|two))'),
      count_ao3_hc=1*str_count(hc,'(ao|oriented)\\s?x?\\s?(3|three)'),
      count_exf=1*str_count(discharge_disposition,'extend|servic'),
      count_hosp=1*str_count(discharge_disposition,'hospice|expir'),
      count_home=1*str_count(discharge_disposition,'home'),
      count_homeless=1*str_count(hc,'homeless|shelter'),
      count_schiz=1*str_count(hc,'schizo|delusion|disorganiz|cataton'),
      count_park=1*str_count(hc,'parkins|dopa|duopa|rytary|sinemet'),
      count_alz=1*str_count(hc,'alzh|brexpip|donepe|galant|memant|rivastig|aricept|exelon|razadyne'),
      count_manic=1*str_count(hc,'manic|mania|bipol|lithium|lumateper|caplyta|idone|latuda|depakote|abilify|saphris|lamictal|aripipr|lamotrig'),
      count_enceph=1*str_count(hc,'enceph'),
      count_nsurg=1*str_count(hc,'neurosurg|craniot'),
      count_psych=1*str_count(hc,'psychiatr'),
      count_geri=1*str_count(hc,'geriatr'),
      count_pal=1*str_count(hc,'paliat'),
      count_inf=1*str_count(hc,'antibiot|bacteremi|mssa|mrsa|sepsi'),
      count_psych_med=1*str_count(hc,'haloperidol|haldol|olanz|symbyax|precedex|dexmedet|seroquel|quetiapine'),
      count_ciwa=1*str_count(hc,'ciwa|alcoho|withdraw|overdos|detox|tremens'),
      count_hep=1*str_count(hc,'hepatit|hepatol|ascit|jaund|cirrh|varices|meld|portal'),
      count_tox=1*str_count(hc,'toxic'),
      count_los=los                          # added 1/2/23
    ) 
  
  if (!mem_save){
    cat('Creating nursing counts.\n')
    x <- x %>% left_join(read_csv(file.path(path,'data_in','NursingSummariesWithMRN_ID.csv')) %>% 
                           select(date_nurs=note_dt,mrn,note_nurs=note_txt) %>%
                           mutate(mrn=as.numeric(mrn)),
                         by='mrn',relationship='many-to-many') %>%
      mutate(date_nurs=ymd(date_nurs)) %>%
      rowwise() %>%
      mutate(note_overlap=if_else(date_nurs >= date_adm && date_nurs <= date_dc,1,0),
             note_nurs=if_else(note_overlap == 1,note_nurs,NA)) %>%
      ungroup() %>%
      group_by(id) %>%
      mutate(note_nurs=paste(note_nurs,collapse=' '),
             n_nurs=sum(note_overlap,na.rm=TRUE)) %>%
      ungroup() %>%
      select(-note_overlap,-date_nurs) %>%
      distinct() %>%
      mutate(
        count_nurse_del=1*str_count(note_nurs,'deliri| cam |cows'),
        count_nurse_conf_ms=1*str_count(note_nurs,'confus|disorient|waxing|sundowni|sun downi|restrain|halluc'),
        count_nurse_ao0_ms=1*str_count(note_nurs,'((ao|oriented)\\s?x?\\s?(0|zero))|((ao|oriented)\\s?x?\\s?(1|one))|((ao|oriented)\\s?x?\\s?(2|two))'),
        count_nurse_ao3_ms=1*str_count(note_nurs,'(ao|oriented)\\s?x?\\s?(3|three)'),
        count_nurse_pysch_med=1*str_count(note_nurs,'haloperidol|haldol|olanz|symbyax|precedex|dexmedet|seroquel|quetiapine'),
        count_nurse_bp_med=1*str_count(note_nurs,'lithium|lumateper|caplyta|idone|latuda|depakote|abilify|saphris|lamictal|aripipr|lamotrig'),
        count_nurse_alz_med=1*str_count(note_nurs,'brexpip|donepe|galant|memant|rivastig|aricept|exelon|razadyne'),
        count_nurse_wd=1*str_count(note_nurs,'ciwa|alcoho|withdraw|overdos|detox|tremens'),
        count_nurse_jaund=1*str_count(note_nurs,'ascit|jaund|cirrh|varices|meld')
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
      left_join(read_csv(file.path('D:\\Dropbox\\embeddings\\delirium',
                                   'data_in',
                                   'NursingSummariesWithMRN_ID.csv')) %>% 
                  select(date_nurs=note_dt,mrn,note_nurs=note_txt) %>%
                  mutate(mrn=as.numeric(mrn)),
                by='mrn',relationship='many-to-many') %>%
      mutate(date_nurs=ymd(date_nurs)) %>%
      rowwise() %>%
      mutate(note_overlap=if_else(date_nurs >= date_adm && date_nurs <= date_dc,1,0),
             note_nurs=if_else(note_overlap == 1,note_nurs,NA)) %>%
      ungroup() %>%
      group_by(id) %>%
      mutate(note_nurs=paste(note_nurs,collapse=' '),
             n_nurs=sum(note_overlap,na.rm=TRUE)) %>%
      ungroup() %>%
      select(-note_overlap,-date_nurs) %>%
      distinct() %>%
      mutate(
        count_nurse_del=1*str_count(note_nurs,'deliri| cam |cows'),
        count_nurse_conf_ms=1*str_count(note_nurs,'confus|disorient|waxing|sundowni|sun downi|restrain|halluc'),
        count_nurse_ao0_ms=1*str_count(note_nurs,'((ao|oriented)\\s?x?\\s?(0|zero))|((ao|oriented)\\s?x?\\s?(1|one))|((ao|oriented)\\s?x?\\s?(2|two))'),
        count_nurse_ao3_ms=1*str_count(note_nurs,'(ao|oriented)\\s?x?\\s?(3|three)'),
        count_nurse_pysch_med=1*str_count(note_nurs,'haloperidol|haldol|olanz|symbyax|precedex|dexmedet|seroquel|quetiapine'),
        count_nurse_bp_med=1*str_count(note_nurs,'lithium|lumateper|caplyta|idone|latuda|depakote|abilify|saphris|lamictal|aripipr|lamotrig'),
        count_nurse_alz_med=1*str_count(note_nurs,'brexpip|donepe|galant|memant|rivastig|aricept|exelon|razadyne'),
        count_nurse_wd=1*str_count(note_nurs,'ciwa|alcoho|withdraw|overdos|detox|tremens'),
        count_nurse_jaund=1*str_count(note_nurs,'ascit|jaund|cirrh|varices|meld')
      )
    
    cat('Merging tables.\n')
    x <- x %>%
      left_join(read_rds(file.path(path,'data_tmp','raw_counts.rds')),
                by='id')
    
  }
  
  return(x)
  
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

chunk <- function(x,n=4096){
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