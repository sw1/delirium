import os
import re
import subprocess
import itertools
import pandas as pd
from lf_functions import sigfigs, read_data


def run_from_cp(run, n_epoch = 1, reduction_factor = 4, sort_by='b_acc'):
    
    script = "lf_train.py"  
    work_dir = "/shared/anesthesia/wolosomething/delirium/cleanrun_01"  
    sweep_path = "/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/sweep"

    label = 'pseudo'
    fr = 100

    df = pd.read_csv(os.path.join(sweep_path,'run_cw_' + str(run-1) + '.csv')).sort_values(by=sort_by,ascending=False)   
    top_n = len(df) // (reduction_factor * 2)

    df_true = df[df['filter_keywords'] == True].head(top_n)
    df_false = df[df['filter_keywords'] == False].head(top_n)
    df = pd.concat([df_true, df_false])
    df.reset_index(drop=True, inplace=True)

    for i in range(len(df)):

        out_dir = os.path.join(work_dir,'longformer','out','sweep','run_cw_' + str(run))

        try:
            path = os.path.join(sweep_path,'run_cw_' + str(run-1),df['path'][i])
            cp_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            cp_folders = [f for f in cp_folders if re.match(r'checkpoint-\d+', f)]
            cp_folders = sorted(cp_folders, key=lambda f: int(f.split('-')[1]))
            cp_folder = cp_folders[-1]
        except IndexError:
            path = os.path.join(sweep_path,'run_cw_' + str(run-2),df['path'][i])
            cp_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            cp_folders = [f for f in cp_folders if re.match(r'checkpoint-\d+', f)]
            cp_folders = sorted(cp_folders, key=lambda f: int(f.split('-')[1]))
            cp_folder = cp_folders[-1] 

        path_cp = os.path.join(path,cp_folder)

        filter_keywords = df['filter_keywords'][i]
        th = df['th'][i]
        lr = df['lr'][i]
        w_decay = df['w_decay'][i]
        n_batch = df['eff_n_batch'][i]
        lab_smooth = df['lab_smooth'][i]
        class_weighting = df['cw'][i]

        folder_name = ('fkw' + str(int(filter_keywords)) +
                       '_th' + str(th) + 
                       '_lr' + sigfigs(lr,1) +
                       '_wd' + sigfigs(w_decay,1) + 
                       '_nb' + str(n_batch) + 
                       '_ls' + sigfigs(lab_smooth,1) +
                       '_cw' + str(int(class_weighting * 100))
        )

        try:
            os.makedirs(os.path.join(out_dir,folder_name))
        except FileExistsError:
            print(f"\nTrial output exists for {folder_name}. Moving to next trial.\n")
            continue

        n_notes = len(read_data(os.path.join(work_dir,'longformer','data','tbl.csv.gz'),
                            exp=label,th=th,fr=fr)['train']['text'])

        n_grad = 2 if n_batch == 64 else 1
        n_batch = 32 if n_batch == 64 else n_batch
        cw = False if class_weighting == 0.5 else True

        n_steps_per_epoch = int(n_notes * n_epoch / n_batch / n_grad)
        log_steps = int(n_steps_per_epoch * 0.05)

        command = [
            "python", script,
            "--sweep", "True",
            "--testing", "False",
            "--load_cp", path_cp,
            "--seed", "215",
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
            "--class_weights", str(cw),
            "--class_weighting", str(class_weighting),
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
            "--wandb_pn", "sweep_run_cw_" + str(run),
            "--final_sweep", "False"
        ]

        print(f"\nRunning trial {folder_name}.")
        subprocess.run(command)

run_from_cp(run=1,n_epoch=1,reduction_factor=4,sort_by='b_acc')
