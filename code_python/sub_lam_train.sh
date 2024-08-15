#!/bin/bash

WORK_DIR="/shared/anesthesia/wolosomething/delirium/cleanrun_01"
SCRIPT="$WORK_DIR/lam_train.py"
CFG="$WORK_DIR/acc_cfg.yaml"

accelerate launch --config_file $CFG --main_process_port 1344 $SCRIPT 

exit 0