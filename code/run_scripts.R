pacman::p_load(tidyverse)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox'
}

ns <- 5:9
fns <- list.files(file.path(path,'code'))
idx <- unlist(sapply(sprintf('^%s_',ns), function(x) which(str_detect(fns,x))))
fns <- fns[idx]
paths <- file.path(path,'code',fns)

for (p in paths){
  cat(sprintf('\n\n\nRunning script %s.\n\n\n',p))
  source(p)
}