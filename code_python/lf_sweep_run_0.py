import os
import subprocess
import shutil
import itertools
import pandas as pd
from lf_functions import sigfigs, read_data


def sweep(run, n_epoch = 0.4):
    
    script = "lf_train.py"  
    work_dir = "/shared/anesthesia/wolosomething/delirium/cleanrun_01"  

    label = 'pseudo'
    fr = 100

    tune_grid = {'filter_keywords': [True,False],
                 'th': [70,80,90],
                 'lr': [2e-6,8e-6],
                 'w_decay': [0.1], #[0.1,0.01],
                 'n_batch': [8,16,32],
                 'lab_smooth': [0.0],
                 'class_weighting': [0.05,0.25,0.5,0.75,0.95],
                 }

    all_combinations = list(itertools.product(*tune_grid.values()))
    tune_grid = pd.DataFrame(all_combinations, columns=tune_grid.keys())
    tune_grid = tune_grid.sample(frac=1).reset_index(drop=False)

    for i in range(len(tune_grid)):

        out_dir = os.path.join(work_dir,'longformer','out','sweep','run_' + run)

        filter_keywords = tune_grid['filter_keywords'][i]
        th = tune_grid['th'][i]
        lr = tune_grid['lr'][i]
        w_decay = tune_grid['w_decay'][i]
        n_batch = tune_grid['n_batch'][i]
        lab_smooth = tune_grid['lab_smooth'][i]
        class_weighting = tune_grid['class_weighting'][i]

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

        print(f"\nTune grid for iteration {i}.\n")
        print(tune_grid.iloc[i])

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
            "--warmup_ratio", "0.05",
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
            "--wandb_pn", "sweep_run_" + run,
        ]

        print(f"\nRunning trial {folder_name}.")
        subprocess.run(command)
    
    
sweep(n_epoch = 0.4, run = 'cw_0')
