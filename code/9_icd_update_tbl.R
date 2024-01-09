library(tidyverse)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

tbl <- read_csv(file.path(path,'to_python','tbl_to_python_updated.csv.gz'))

subsets <- c('icd','sub_chapter','major')
mods <- c('rf','tree')

for (ss in subsets){
    for (mod in mods){
        
      new_labels <- read_csv(file.path(path,'data_in',
                                       sprintf('labels_%s_count_del_%s.csv.gz',
                                               mod,ss)))
      
      tbl_update <- new_labels %>% 
        select(id,label_tree)
      
      if (mod == 'tree'){
        tbl_update <- tbl_update %>%
          mutate(label_icd = label_tree) %>%
          select(-label_tree) %>%
          left_join(tbl %>% select(-label_icd),by='id') 
        
        write_csv(tbl_update,
                  file.path(
                    path,'to_python',
                    sprintf('tbl_to_python_updated_count_del_%s_%s.csv.gz',
                            mod,ss)),
                  col_names=TRUE)
        
        cat(do.call(sprintf, 
                    c(fmt='Predicted labels (%s, %s): \n0: %s\n1: %s.\n', 
                      as.list(c(mod,ss,table(tbl_update$label_icd))))))
      }
      
      if (mod == 'rf'){
        tbl_update <- tbl_update %>%
          mutate(label_icd = label_tree) %>%
          select(-label_tree) %>%
          left_join(tbl %>% select(-label_icd),by='id') 
        
        for (p in c(0.5,0.6,0.65,.7)){
          tbl_tmp <- tbl_update %>%
            mutate(label_icd=if_else(label_icd >= p,1,0))
          
          write_csv(
            tbl_tmp,
            file.path(path,'to_python',
                      sprintf('tbl_to_python_updated_count_del_%s%s_%s.csv.gz',
                              mod,p*100,ss)),
                    col_names=TRUE)
          
          cat(do.call(
            sprintf,c(fmt='Predicted labels (%s, %s, p=%s): \n0: %s\n1: %s.\n',
                      as.list(c(mod,ss,p,table(tbl_tmp$label_icd))))))
        }
      }
      
      cat(do.call(
        sprintf,c(fmt='Dimensions of table (%s, %s): %s x %s.\n',
                  as.list(c(mod,ss,dim(tbl_update))))))
  }
}