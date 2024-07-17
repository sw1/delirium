#!/bin/bash

WORK_DIR="/shared/anesthesia/wolosomething/delirium/cleanrun_01"
SCRIPT="$WORK_DIR/lam_train.py"
CFG="$WORK_DIR/acc_cfg.yaml"

accelerate launch --config_file $CFG --main_process_port 4124 \
    $SCRIPT \
    --seed 14231 \
    --label 'pseudo' \
    --overwrite_prompt True \
    --threshold 90 \
    --fraction 100 \
    --pipeline 1 \
    --seq_len 8192 \
    --n_grad_accum 4 \
    --n_grad_accum_eval 4 \
    --n_batch 2 \
    --n_batch_eval 2 \
    --n_train_epochs 4 \
    --lr 2e-06 \
    --warmup_ratio 0.05 \
    --w_decay 0.01 \
    --f_log_steps 100 \
    --save_multiplier 2 \
    --do_hidden 0.1 \
    --do_class 0.1 \
    --label_smoothing 0.0 \
    --upsample False \
    --class_weights True \
    --class_weighting 0.05 \
    --filter_keywords False \
    --group_by_len True \
    --pad_max_len False \
    --use_collator True \
    --n_cores 16 \
    --num_labels 2 \
    --bf16 True \
    --mod_load_in_8bit False \
    --mod_torch_dtype 'bfloat16' \
    --bnb_load_in_4bit True \
    --bnb_4bit_quant_type 'nf4' \
    --bnb_4bit_use_double_quant True \
    --bnb_4bit_compute_dtype 'bfloat16' \
    --bnb_4bit_quant_storage_dtype 'bfloat16' \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_r 8 \
    --lara_bias 'none' \
    --lora_target_modules 'all-linear' \
    --mod 8 \
    --input_table 'tbl_allnotes_trimmedlen.csv.gz' \
    --work_dir "$WORK_DIR/llama" \
    #--n_steps_testing 10 \
    #--f_subset_data 0.05 \
    #--folder_suffix 'testing1234' \
    #--out_dir 'shared/anesthesia/wolosomething/scratch' \


exit 0
