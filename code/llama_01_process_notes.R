pacman::p_load(tidyverse,textstem,tidytext,lubridate,tm,glue,doParallel)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
  all_cores <- 4
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
  all_cores <- 8
}
source(file.path(path,'code','fxns.R'))

cl <- makeCluster(all_cores)
registerDoParallel(cl)

years <- list.files(file.path(path,'data_in'),pattern='notes_[0-9]+\\.csv.gz')
years <- unique(str_extract(years,'[0-9]+'))


res <- foreach(i = seq_along(years), 
               .combine=rbind, 
               .export=c('years'),
               .packages=c('tidyverse','glue')) %dopar% {
                 
                 read_rds(file.path(path,'data_tmp',
                                    glue('notes_filt_{years[i]}.rds'))) %>%
                   mutate(note = tolower(note),
                          note = str_replace_all(note, '\\r|\\n', ' '),
                          note = aox(note),
                          note = str_replace_all(note, 
                                                 glue("([^a-z '.!,?]|(?<![a-z])",
                                                      "[[:punct:]](?![a-z]))"),
                                                 ''),
                          note = str_replace_all(note,
                                                 '.*affiliation bidmc',''),
                          note = str_squish(note)) %>%
                   write_rds(file.path(path,'data_tmp',
                                       glue('notes_proc_{years[i]}.rds')))
}

stopCluster(cl)
