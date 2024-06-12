#!/bin/bash

WORK_DIR="/shared/anesthesia/wolosomething/delirium/cleanrun_01"
TBL_DIR="$WORK_DIR/longformer/data"
OUT_DIR="$WORK_DIR/longformer/out/finetune"

s1="3466"
s2="12433"

cd $WORKDIR

TBLS=($(ls ${TBL_DIR}/*.csv.gz))

prefix="tbl_to_python_expertupdate_"
suffix=".csv.gz"

for f in ${TBLS[@]}
do
    folder_name=${f#$prefix}
    folder_name=${folder_name%$suffix}
    if [ -d "$OUT_DIR/$folder_name" ]; then
        echo -ne "$folder_name exists\n"
    else
        echo -ne "running script for $f\n"
        python lf_finetune.py $f 1 $s1 $s2 1
    fi
done

exit 0