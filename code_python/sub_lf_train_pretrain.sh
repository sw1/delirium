#!/bin/bash

WORK_DIR="/shared/anesthesia/wolosomething/delirium/cleanrun_01"
SCRIPT="$WORK_DIR/lf_train.py"

python $SCRIPT \
    --seed 14231 \
    --train_method 'pretrain' \
    --label 'icd' \
    --override_prompt True \
    --threshold 70 \
    --fraction 100 \
    --pipeline 1 \
    --seq_len 4096 \
    --n_grad_accum 8 \
    --n_grad_accum_eval 1 \
    --n_batch 8 \
    --n_batch_eval 64 \
    --n_train_epochs 15 \
    --lr 3e-5 \
    --warmup_ratio 0.07 \
    --n_cycles 2 \
    --w_decay 0.01 \
    --f_log_steps 0.5 \
    --save_multiplier 5 \
    --do_hidden 0.1 \
    --do_class 0.1 \
    --label_smoothing 0.0 \
    --upsample False \
    --class_weights False \
    --filter_keywords False \
    --group_by_len True \
    --pad_max_len False \
    --use_collator True \
    --n_cores 16 \
    --num_labels 2 \
    --input_table 'tbl.csv.gz' \
    --work_dir "$WORK_DIR/longformer" \
    #--n_steps_testing 10 \
    #--f_subset_data 0.05 \
    #--folder_suffix 'testing1234' \
    #--out_dir 'shared/anesthesia/wolosomething/scratch' \


exit 0
