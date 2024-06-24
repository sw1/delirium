#!/bin/bash

WORK_DIR="/shared/anesthesia/wolosomething/delirium/cleanrun_01"
TBL_DIR="$WORK_DIR/longformer/data"
OUT_DIR="$WORK_DIR/longformer/out/finetune"

script="$WORK_DIR/lf_finetune.py"

s=2261

cd $WORK_DIR

for l in "pseudo" "icd" "full" "only"
do
    if [ $l == "pseudo" ]; then
        for t in 70 60 80 90
        do
            if [ $t == 70 ]; then
                for f in 100 75 50 35 20
                do
                    if [ $f == 100 ]; then
                        #for p in 1 2 3
                        for p in 1
                        do
                            folder_name="fit_${l}_chunked_th${t}_fr${f}_pl${p}_s${s}"
                            if [ -d "$OUT_DIR/$folder_name/final_model_finetune" ]; then
                                echo -ne "$folder_name exists for pipeline 1\n"
                            else
                                echo -ne "running $folder_name\n"
                                python $script -l $l -t $t -f $f -p $p -s $s
                            fi
                        done
                    else
                        folder_name="fit_${l}_chunked_th${t}_fr${f}_pl1_s${s}"
                        if [ -d "$OUT_DIR/$folder_name/final_model_finetune" ]; then
                            echo -ne "$folder_name exists for pipeline 1\n"
                        else
                            echo -ne "running $folder_name\n"
                            python $script -l $l -t $t -f $f -s $s
                        fi
                    fi
                done
            else
                folder_name="fit_${l}_chunked_th${t}_fr100_pl1_s${s}"
                if [ -d "$OUT_DIR/$folder_name/final_model_finetune" ]; then
                    echo -ne "$folder_name exists for pipeline 1\n"
                else
                    echo -ne "running $folder_name\n"
                    python $script -l $l -t $t -s $s
                fi
            fi
        done
    else
        folder_name="fit_${l}_chunked_pl1_s${s}"
        if [ -d "$OUT_DIR/$folder_name/final_model_finetune" ]; then
            echo -ne "$folder_name exists for pipeline 1\n"
        else
            echo -ne "running $folder_name\n"
            python $script -l $l -s $s
        fi
    fi
done


exit 0
