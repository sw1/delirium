pacman::p_load(tidyverse,glue,gtsummary,flextable,icd.data,tidymodels,
               rpart,rpart.plot,officer,gridExtra,ggrepel)


if (Sys.info()['login'] == 'sw1'){
  path <- 'D:\\Dropbox\\embeddings\\delirium'
}
if (Sys.info()['login'] == 'swolosz1'){
  path <- 'C:\\Users\\swolosz1\\Dropbox\\embeddings\\delirium'
}
source(file.path(path,'code','fxns.R'))

calc_m <- function(x1,x2,y1,y2) (y2-y1)/(x2-x1)


trend_plot <- function(trends,ids,filt=NULL,
                       ndiff=1,
                       lab_size=1.75,
                       window=NULL,
                       span=0.15,
                       top_n_pos=2,top_n_neg=2,
                       n_wrap=20){
  
  if (is.null(window)) window <- 100*ndiff
  
  dat <- trends %>% 
    filter(id %in% ids) %>%
    mutate(id = paste0('Obs. ',dense_rank(id)))
  
  dat <- dat %>%
    mutate(subseqs=substr(text,
                          nchar(text)-window,
                          nchar(text)),
           subseqs=str_wrap(subseqs,width=n_wrap)) %>%
    arrange(id,run,i) %>% 
    mutate(m=0) %>%
    group_by(run,id) %>%
    mutate(smoothed = loess(scores ~ i,span=span)$fitted) %>%
    ungroup()
  
  dat_tmp <- tibble()
  for (obs in unique(dat$id)){
    for (r in unique(dat$run)){
      dat_obs <- dat %>% filter(id == obs,run == r)
      for (row in 2:nrow(dat_obs)){
        if (row <= ndiff){
          dat_obs[row,]$m <- calc_m(x1=dat_obs[1,]$i,x2=dat_obs[row,]$i,
                                    y1=dat_obs[1,]$scores,y2=dat_obs[row,]$scores) 
        }else{
          dat_obs[row,]$m <- calc_m(x1=dat_obs[row - ndiff,]$i,x2=dat_obs[row,]$i,
                                    y1=dat_obs[row - ndiff,]$scores,y2=dat_obs[row,]$scores)
        }
      }
      dat_tmp <- dat_tmp %>% bind_rows(dat_obs)
    }
  }
  dat <- dat_tmp
  
  dat <- dat %>%
    group_by(run,id) %>%
    mutate(labels=factor(labels,levels=c(0,1),ordered=TRUE),
           rank_pos=dense_rank(desc(m)),
           rank_neg=dense_rank(m))
  
  if (!is.null(filt)){
    filt <- paste0(filt,collapse = '|')
    dat <- dat %>%
      mutate(subseqs=str_replace_all(subseqs,filt,''))
  }
  
  repel_tbl <- dat %>%
    group_by(run,id) %>%
    filter(rank_pos <= top_n_pos | rank_neg <= top_n_neg) %>%
    mutate(direction=if_else(m > 0,'pos','neg')) %>%
    ungroup()

  p <- dat %>%
    ggplot(aes(i,smoothed,color=labels)) +
    geom_hline(yintercept=.5,linetype=3,color='black',alpha=.9) +
    geom_point(aes(i,scores),color='gray',alpha=.7) +
    geom_line(linewidth=1.5,alpha=.5) +
    geom_label_repel(data=repel_tbl,
                     aes(i,smoothed,label=subseqs,fill=direction,color=direction),
                     size=lab_size,alpha=.5,color='black',
                     min.segment.length=0.0,
                     # nudge_x = 5, 
                     # nudge_y = -.5,
                     direction='both',
                     ylim=c(-1,1),
                     max.time=5,
                     max.iter=100000,
                     # force_pull = -5,
                     # force = 15,
                     max.overlaps = 10) +
    facet_grid(id~run) +
    scale_fill_manual(values=c('lightblue','orange')) +
    scale_color_manual(values=c('blue','red'),drop=FALSE) +
    ylim(-1,1) +
    labs(x='Note Subsequence Index',y='Prediction Score',color='') +
    theme_classic() +
    scale_y_continuous(limits = c(-1, 1), breaks = c(1, 0.5, 0)) + 
    theme(legend.position='none',
          panel.border = element_rect(color='black',fill=NA,linewidth=0.5),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank())
  
  return(p)
}

trends1 <- read_csv(file.path(path,'from_python','fkw1_th70_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_expert.csv.gz')) %>%
  mutate(run = 'Threshold 70')
trends2 <- read_csv(file.path(path,'from_python','fkw1_th80_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_expert.csv.gz')) %>%
  mutate(run = 'Threshold 80')
trends3 <- read_csv(file.path(path,'from_python','fkw1_th90_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_expert.csv.gz')) %>%
  mutate(run = 'Threshold 90')
trends <- bind_rows(trends1,trends2,trends3)

mms <- trends %>% 
  group_by(id,run) %>% 
  filter(i == max(i)) %>% 
  group_by(id) %>% 
  filter(!(sum(preds) %in% c(0,3))) %>% 
  select(id) %>%
  distinct() %>%
  pull(id) 

# samps <- c(311151,517814,681741)
samps <- c(4937,609056,86512)
(fig <- trend_plot(trends,samps,lab_size=1.75,top_n_pos=2,top_n_neg=2,
                   ndiff=5,n_wrap=20,window=150)) 

ggsave(file.path(path,'figs','trends_p708090.png'), 
       plot = fig, width = 15, height = 7.5, dpi = 300)


trends1 <- read_csv(file.path(path,'from_python','fkw1_th90_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_expert.csv.gz')) %>%
  mutate(run = 'FT')
trends2 <- read_csv(file.path(path,'from_python','fkw1_th90_fr100_pl2_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_expert.csv.gz')) %>%
  mutate(run = 'FT + PT')
trends <- bind_rows(trends1,trends2)

mms <- trends %>% 
  group_by(id,run) %>% 
  filter(i == max(i)) %>% 
  group_by(id) %>% 
  filter(!(sum(preds) %in% c(0,2))) %>% 
  pull(id) %>% 
  unique()

# samps <- c(446504,517814,663439)
samps <- c(26483,663617 ,664525)
(fig <- trend_plot(trends,samps,lab_size=1.75,top_n_pos=2,top_n_neg=2,
                   ndiff=5,n_wrap=30,window=150)) 

ggsave(file.path(path,'figs','trends_p90_123.png'),
       plot = fig, width = 10, height = 5, dpi = 300)



trends1 <- read_csv(file.path(path,'from_python','fkw1_th90_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_expert.csv.gz')) %>%
  mutate(run = 'Longformer')
trends2 <- read_csv(file.path(path,'from_python','fkw1_th90_fr100_pl1_lr5.0e-05_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_expert_llama.csv.gz')) %>%
  mutate(run = 'Llama')
trends <- bind_rows(trends1,trends2)

mms <- trends %>% 
  group_by(id,run) %>% 
  filter(i == max(i)) %>% 
  group_by(id) %>% 
  filter(!(sum(preds) %in% c(0,2))) %>% 
  pull(id) %>% 
  unique()

samps <- c(4937 ,609056,646081)
(fig <- trend_plot(trends,samps,lab_size=1.75,top_n_pos=2,top_n_neg=2,
                   ndiff=5,n_wrap=30,window=150)) 


ggsave(file.path(path,'figs','trends_plflam.png'),
       plot = fig, width = 10, height = 5, dpi = 300)






trends1 <- read_csv(file.path(path,'from_python','fkw1_th70_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_unlabeled.csv.gz')) %>%
  mutate(run = 'Threshold 70')
trends2 <- read_csv(file.path(path,'from_python','fkw1_th80_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_unlabeled.csv.gz')) %>%
  mutate(run = 'Threshold 80')
trends3 <- read_csv(file.path(path,'from_python','fkw1_th90_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_unlabeled.csv.gz')) %>%
  mutate(run = 'Threshold 90')
trends <- bind_rows(trends1,trends2,trends3)

mms <- trends %>% 
  group_by(id,run) %>% 
  filter(i == max(i)) %>% 
  group_by(id) %>% 
  filter(!(sum(preds) %in% c(0,3))) %>% 
  select(id) %>%
  distinct() %>%
  pull(id) 

samps <- c(15926, 468404, 10137  )
fig <- trend_plot(trends,samps,lab_size=1.75,top_n_pos=2,top_n_neg=2,
                   ndiff=5,n_wrap=30,window=150)  + 
  scale_color_manual(values=c('black','black','black'),drop=FALSE)

ggsave(file.path(path,'figs','trends_p708090_unlab.png'), 
       plot = fig, width = 15, height = 7.5, dpi = 300)


trends1 <- read_csv(file.path(path,'from_python','fkw1_th90_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_unlabeled.csv.gz')) %>%
  mutate(run = 'FT')
trends2 <- read_csv(file.path(path,'from_python','fkw1_th90_fr100_pl2_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_unlabeled.csv.gz')) %>%
  mutate(run = 'FT + PT')
trends <- bind_rows(trends1,trends2)

mms <- trends %>% 
  group_by(id,run) %>% 
  filter(i == max(i)) %>% 
  group_by(id) %>% 
  filter(!(sum(preds) %in% c(0,2))) %>% 
  pull(id) %>% 
  unique()


samps <- c(178769,358399,370298)
fig <- trend_plot(trends,samps,lab_size=1.75,top_n_pos=2,top_n_neg=2,
                   ndiff=5,n_wrap=30,window=150)  + 
  scale_color_manual(values=c('black','black','black'),drop=FALSE)

ggsave(file.path(path,'figs','trends_p90_123_unlab.png'),
       plot = fig, width = 10, height = 5, dpi = 300)



trends1 <- read_csv(file.path(path,'from_python','fkw1_th90_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_unlabeled.csv.gz')) %>%
  mutate(run = 'Longformer')
trends2 <- read_csv(file.path(path,'from_python','fkw1_th90_fr100_pl1_lr5.0e-05_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo_unlabeled_llama.csv.gz')) %>%
  mutate(run = 'Llama')
trends <- bind_rows(trends1,trends2)

mms <- trends %>% 
  group_by(id,run) %>% 
  filter(i == max(i)) %>% 
  group_by(id) %>% 
  filter(!(sum(preds) %in% c(0,2))) %>% 
  pull(id) %>% 
  unique()

samps <- c(196802,183001,518557)
fig <- trend_plot(trends,samps,lab_size=1.75,top_n_pos=2,top_n_neg=2,
                   ndiff=5,n_wrap=30,window=150)  + 
  scale_color_manual(values=c('black','black','black'),drop=FALSE)

ggsave(file.path(path,'figs','trends_plflam_unlab.png'),
       plot = fig, width = 10, height = 5, dpi = 300)
