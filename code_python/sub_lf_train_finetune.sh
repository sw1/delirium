#!/bin/bash

WORK_DIR="/shared/anesthesia/wolosomething/delirium/cleanrun_01"
SCRIPT="$WORK_DIR/lf_train.py"

python $SCRIPT \
    --seed 14231 \
    --train_method 'finetune' \
    --label 'icd' \
    --override_prompt True \
    --threshold 70 \
    --fraction 100 \
    --pipeline 1 \
    --seq_len 4096 \
    --n_grad_accum 2 \
    --n_grad_accum_eval 1 \
    --n_batch 32 \
    --n_batch_eval 64 \
    --n_train_epochs 4 \
    --lr 8e-6 \
    --warmup_ratio 0.07 \
    --n_cycles 0.5 \
    --w_decay 0.01 \
    --f_log_steps 0.1 \
    --save_multiplier 2 \
    --do_hidden 0.1 \
    --do_class 0.1 \
    --label_smoothing 0.0 \
    --upsample True \
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
