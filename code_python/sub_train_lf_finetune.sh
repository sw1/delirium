#!/bin/bash

WORK_DIR="/shared/anesthesia/wolosomething/delirium/cleanrun_01"
SCRIPT="$WORK_DIR/lf_train.py"

python $SCRIPT \
    --seed 14231 \
    --train_method 'finetune' \
    --label 'pseudo' \
    --overwrite_prompt True \
    --threshold 70 \
    --fraction 100 \
    --pipeline 1 \
    --seq_len 4096 \
    --n_grad_accum 1 \
    --n_grad_accum_eval 1 \
    --n_batch 16 \
    --n_batch_eval 64 \
    --n_train_epochs 4 \
    --lr 8e-06 \
    --warmup_ratio 0.05 \
    --n_cycles 0.5 \
    --w_decay 0.01 \
    --f_log_steps 0.1 \
    --save_multiplier 2 \
    --do_hidden 0.1 \
    --do_class 0.1 \
    --label_smoothing 0.0 \
    --upsample False \
    --class_weights True \
    --filter_keywords False \
    --group_by_len True \
    --pad_max_len False \
    --use_collator True \
    --n_cores 16 \
    --num_labels 2 \
    --input_table 'tbl.csv.gz' \
    --work_dir "$WORK_DIR/longformer" \
    --testing False 

exit 0
