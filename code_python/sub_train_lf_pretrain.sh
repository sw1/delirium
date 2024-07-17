#!/bin/bash

WORK_DIR="/shared/anesthesia/wolosomething/delirium/cleanrun_01"
SCRIPT="$WORK_DIR/lf_train.py"
CFG="$WORK_DIR/acc_cfg.yaml"

python $SCRIPT \
    --seed 14231 \
    --train_method 'pretrain' \
    --label 'full' \
    --overwrite_prompt True \
    --pipeline 2 \
    --seq_len 4096 \
    --n_grad_accum 4 \
    --n_grad_accum_eval 4 \
    --n_batch 4 \
    --n_batch_eval 4 \
    --n_train_epochs 15 \
    --lr 3e-5 \
    --warmup_ratio 0.05 \
    --w_decay 0.01 \
    --do_hidden 0.1 \
    --filter_keywords False \
    --group_by_len True \
    --pad_max_len False \
    --use_collator True \
    --n_cores 16 \
    --input_table 'tbl.csv.gz' \
    --work_dir "$WORK_DIR/longformer" \
    --testing False 


exit 0
