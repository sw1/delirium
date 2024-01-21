library(tidyverse)
library(glue)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

chunked <- TRUE

tbl <- read_csv(file.path(path,'to_python','tbl_to_python_updated.csv.gz'))

tbl_chunked <- tbl %>%
  mutate(chunked = if_else(nchar(hpi_hc) > 4096,1,0)) %>%
  rowwise() %>%
  mutate(hpi_hc=if_else(chunked == 1,chunk(hpi_hc),list(hpi_hc))) %>%
  ungroup() %>%
  unnest(hpi_hc) %>%
  select(-chunked) %>%
  filter(nchar(hpi_hc) >= 300)

write_csv(tbl_chunked,
          file.path(path,'to_python','tbl_to_python_updated_chunked.csv.gz'))

# subsets <- c('icd','sub_chapter','major')
# mods <- c('rf','tree')

if (chunked){
  tbl <- tbl_chunked
  tbl_fn <- 'updated_chunked'
}else{
  tbl_fn <- 'updated'
}

fns <- list.files(file.path(path,'data_in'),
                  pattern='^labels_rf',full.names=TRUE)

for (fn in fns){
  
  mod <- NULL
  st <- NULL
  
  fn_split <- str_split(fn,'_|\\.')[[1]]
  fn_split <- fn_split[3:(length(fn_split)-2)]
  fn_split <- fn_split[!(fn_split %in% c('count','del'))]
  
  new_labels <- read_csv(fn)
  
  if (any(str_detect(fn_split,'rf'))) mod <- 'rf'
  if (any(str_detect(fn_split,'tree'))) mod <- 'tree'
  if (any(str_detect(fn_split,'rfst'))) st <- 'self-training'
  if (any(str_detect(fn_split,'th\\d+'))){
    thr <- fn_split[str_detect(fn_split,'th\\d+')]
    thr <- str_match(thr,'th(\\d+)')[,2]
  }
  if (any(str_detect(fn_split,'w\\d+'))){
    w <- fn_split[str_detect(fn_split,'w\\d+')]
    w <- str_match(w,'w(\\d+)')[,2]
  }
  if (any(str_detect(fn_split,'^f+'))){
    ss <- fn_split[str_detect(fn_split,'^f+')]
    ss <- str_match(ss,'^f(.*)')[,2]
  }else{
    ss <- fn_split[2]
  }
  if (any(str_detect(fn_split,'^\\d+'))){
    f <- fn_split[str_detect(fn_split,'^\\d+')]
  }
  
  if (mod == 'tree'){
    tbl_update <- new_labels %>%
      select(id,label_icd=label_tree) %>%
      left_join(tbl %>% select(-label_icd),by='id') 
    
    lab_counts <- table(tbl_update$label_icd)
    count_0 <- lab_counts[1]
    count_1 <- lab_counts[2]
    
    cat(glue('Saving {tbl_fn} mod:{mod} code:{ss}\n\\
  Label 0: {count_0}\n\\
  Label 1: {count_1}\n\n'))
    
    fn_out <- glue("tbl_to_python_{tbl_fn}_tree_s{ss}.csv.gz")
    write_csv(tbl_update,file.path(path,'to_python',fn_out))
  }
  
  if (mod == 'rf'){
    
    if (!is.null(st)){
      
      tbl_update <- new_labels %>%
        rename(label_icd = label) %>%
        left_join(tbl %>% select(-label_icd),by='id') 
      
      if (any(is.na(tbl_update$label_icd))){
        
        tbl_tmp <- tbl_update %>%
          filter(!is.na(label_icd))
        
        lab_counts <- table(tbl_tmp$label_icd)
        count_0 <- lab_counts[1]
        count_1 <- lab_counts[2]
        
        cat(glue('Saving {tbl_fn} mod:{mod} code:{ss} threshold:{thr} \\
weight:{w} n_feature:{f}\n\\
  Label 0: {count_0}\n\\
  Label 1: {count_1}\n\\
  Filtered\n\n'))
        
        fn_out <- glue("tbl_to_python_{tbl_fn}_rfst_s{ss}_t{thr}_w{w}_f{f}_\\
                     filtered.csv.gz")
        write_csv(tbl_tmp,file.path(path,'to_python',fn_out))
        
        tbl_tmp <- tbl_update %>%
          mutate(label_icd=if_else(is.na(label_icd),0,label_icd))
        
        lab_counts <- table(tbl_tmp$label_icd)
        count_0 <- lab_counts[1]
        count_1 <- lab_counts[2]
        
        cat(glue('Saving {tbl_fn} mod:{mod} code:{ss} threshold:{thr} \\
weight:{w} n_feature:{f}\n\\
  Label 0: {count_0}\n\\
  Label 1: {count_1}\n\\
  Zeroed\n\n'))
        
        fn_out <- glue("tbl_to_python_{tbl_fn}_rfst_s{ss}_t{thr}_w{w}_f{f}_\\
                     zeroed.csv.gz")
        write_csv(tbl_tmp,file.path(path,'to_python',fn_out))
        
      }else{
        
        lab_counts <- table(tbl_update$label_icd)
        count_0 <- lab_counts[1]
        count_1 <- lab_counts[2]
        
        cat(glue('Saving {tbl_fn} mod:{mod} code:{ss} threshold:{thr} \\
weight:{w} n_feature:{f}\n\\
  Label 0: {count_0}\n\\
  Label 1: {count_1}\n\n'))
        
        fn_out <- glue("tbl_to_python_{tbl_fn}_rfst_s{ss}_t{thr}_w{w}_f{f}\\
                       .csv.gz")
        write_csv(tbl_update,file.path(path,'to_python',fn_out))
        
      }
      
    }else{
      
      tbl_update <- new_labels %>%
        select(id,label_icd = label_tree) %>%
        left_join(tbl %>% select(-label_icd),by='id') 
      
      for (p in c(0.5,0.6,0.65)){
        tbl_tmp <- tbl_update %>%
          mutate(label_icd=if_else(label_icd >= p,1,0))
        
        lab_counts <- table(tbl_tmp$label_icd)
        count_0 <- lab_counts[1]
        count_1 <- lab_counts[2]
        
        cat(glue('Saving {tbl_fn} mod:{mod} code:{ss} threshold:{p*100} \n\\
  Label 0: {count_0}\n\\
  Label 1: {count_1}\n\n'))
        
        fn_out <- glue("tbl_to_python_{tbl_fn}_rf_s{ss}_t{p*100}.csv.gz")
        write_csv(tbl_tmp,file.path(path,'to_python',fn_out))

      }
    }
  }
}