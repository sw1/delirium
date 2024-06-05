pacman::p_load(tidyverse,glue)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}

fns <- tibble(fn=list.files(file.path(path,'code'))) %>%
  mutate(idx=as.numeric(str_extract(fn,'(\\d+)_.*',group=1))) %>%
  filter(!is.na(idx)) %>%
  arrange(idx) %>%
  filter(idx >= 0)

for (i in 1:nrow(fns)){
  cat(glue('\n\n\nRunning script {fns$fn[i]}.\n\n\n'))
  p <- file.path(path,'code',fns$fn[i])
  env_tmp <- new.env()
  source(p,local=env_tmp,echo=FALSE)
}