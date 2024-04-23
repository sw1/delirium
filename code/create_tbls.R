library(tidyverse)
library(glue)
library(gtsummary)

if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'sw424'){
  path <- 'C:\\Users\\sw424\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

# sample demo/feature table

tbl <- read_rds(file.path(path,'data_in','03_tbl_final.rds'))

tbl %>% 
  mutate(sex=if_else(sex==0,'female','male'),
         label=case_when(
           is.na(label) ~ 'Unlabeled',
           label == 1 ~ 'Label: 1',
           label == 0 ~ 'Label: 0',
           TRUE ~ NA)) %>%
  select(label,service,sex,age,los,num_meds,num_allergies,len_pmhx) %>%
  tbl_summary(by=label,
              statistic=list(all_continuous() ~ '{mean} ({sd})',       
                             all_categorical() ~ '{n} ({p}%)'),   
              digits=all_continuous() ~ 1,                             
              type=all_categorical() ~ 'categorical',                
              label=list(                                           
                label ~ 'Label', 
                age ~ 'Age (Years)',
                sex ~ 'Gender',
                los ~ 'Length of Stay (Days)',
                service ~ 'Service',
                num_meds ~ 'Medications on Admission (Count)',
                num_allergies ~ 'Allergies on Admission (Count)',
                len_pmhx ~ 'Length of Past Medical History'),
              missing_text="Missing") %>%
  add_p(age ~ 'kruskal.test')
