import os
import subprocess
import itertools
import pandas as pd
from lf_functions import sigfigs, read_data

script = "lf_train.py"  
work_dir = "/shared/anesthesia/wolosomething/delirium/cleanrun_01"  
sweep_path = "/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/sweep"
run = 'run_' + str(run)

n_epoch = 1.0
run = 1
reduction_factor = 4

label = 'pseudo'
th = 70
fr = 100
n_notes = len(read_data(os.path.join(work_dir,'longformer','data','tbl.csv.gz'),
                        exp=label,th=th,fr=fr)['train']['text'])


df = pd.read_csv(os.path.join(sweep_path,'run_' + str(run-1),'.csv'), index=False)
top_n = len(df) // reduction_factor

for i in range(top_n):
    
    path = df['path'][i]
    path_cp = os.path.join(path,'model_trainer')
    
    filter_keywords = df['filter_keywords'][i]
    lr = df['lr'][i]
    w_decay = df['w_decay'][i]
    n_batch = df['n_batch'][i]
    lab_smooth = df['lab_smooth'][i]
    
    n_grad = 2 if n_batch == 64 else 1
    n_batch = 32 if n_batch == 64 else n_batch
    
    n_steps_per_epoch = int(n_notes * n_epoch / n_batch / n_grad)
    log_steps = int(n_steps_per_epoch * 0.05)
    
    folder_name = ('fkw' + str(int(filter_keywords)) +
                   '_lr' + sigfigs(lr,1) +
                   '_wd' + sigfigs(w_decay,1) + 
                   '_nb' + str(n_batch) + 
                   '_ns' + sigfigs(lab_smooth,1)
    )
    
    out_dir = os.path.join(work_dir,'longformer','out','sweep','run_' + str(run))

    command = [
        "python", script,
        "--sweep", "True",
        "--testing", "False",
        "--load_cp", path_cp,
        "--seed", "14231",
        "--train_method", "finetune",
        "--label", label,
        "--overwrite_prompt", "False",
        "--threshold", str(th),
        "--fraction", str(fr),
        "--pipeline", "1",
        "--seq_len", "4096",
        "--log_steps", str(log_steps),
        "--n_grad_accum", str(n_grad),
        "--n_grad_accum_eval", "1",
        "--n_batch", str(n_batch),
        "--n_batch_eval", "64",
        "--n_train_epochs", str(n_epoch),
        "--lr", str(lr),
        "--warmup_ratio", "0",
        "--n_cycles", "0.5",
        "--w_decay", str(w_decay),
        "--f_log_steps", "0.1",
        "--save_multiplier", "2",
        "--do_hidden", "0.1",
        "--do_class", "0.1",
        "--label_smoothing", str(lab_smooth),
        "--upsample", "False",
        "--class_weights", "True",
        "--filter_keywords", str(filter_keywords),
        "--group_by_len", "True",
        "--pad_max_len", "False",
        "--use_collator", "True",
        "--n_cores", "16",
        "--num_labels", "2",
        "--input_table", "tbl.csv.gz",
        "--out_dir", out_dir,
        "--work_dir", os.path.join(work_dir,"longformer"),
        "--folder_fn", folder_name,
        "--wandb_pn", "sweep_run_" + str(run),
    ]
    
    if os.path.exists(os.path.join(out_dir,folder_name)):
        print(f"\nTrial output exists for {folder_name}. Moving to next trial.\n")
        next
    else:
        print(f"\nRunning trial {folder_name}.")
        os.makedirs(os.path.join(out_dir,folder_name)) 
        subprocess.run(command)
    
