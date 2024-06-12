#!/bin/bash

WORK_DIR="/shared/anesthesia/wolosomething/delirium/cleanrun_01"
TBL_DIR="$WORK_DIR/longformer/data"
OUT_DIR="$WORK_DIR/longformer/out/finetune"

s1="3466"
s2="12433"
optimal_table="tbl_to_python_expertupdate_chunked_rfst_majvote_th70_ns0.csv.gz"

script="$WORK_DIR/lf_finetune.py"

cd $WORKDIR

TBLS=()
for file in ${TBL_DIR}/*.csv.gz
do
    TBLS+=($(basename $file))
done

prefix="tbl_to_python_expertupdate"
suffix=".csv.gz"

for f in ${TBLS[@]}
do
    folder_name=${f#$prefix}
    folder_name=${folder_name%$suffix}
    folder_name="fit$folder_name"
    if [ -d "$OUT_DIR/$folder_name/final_model_finetune" ]; then
        echo -ne "$folder_name exists for pipeine 1\n"
    else
        echo -ne "running pipeline 1 script for $f\n"
        python $script $f 1 $s1 $s2 1
    fi
    if [ "$f" == "$optimal_table" ]; then
        if [ -d "$OUT_DIR/$folder_name/final_model_pretrain_finetune" ]; then
            echo -ne "$folder_name exists for pipeine 2\n"
        else
            echo -ne "running pipeline 2 script for $f\n"
            python $script $f 2 $s1 $s2 1
        fi
        if [ -d "$OUT_DIR/$folder_name/final_model_token_pretrain_finetune" ]; then
            echo -ne "$folder_name exists for pipeine 3\n"
        else
            echo -ne "running pipeline 3 script for $f\n"
            python $script $f 3 $s1 $s2 1
        fi
    fi
done

exit 0