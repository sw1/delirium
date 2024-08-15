import os
import re
import subprocess
import itertools
import pandas as pd
from lf_functions import sigfigs, read_data


def run_from_cp(n_epoch = 3):
    
    script = "lf_train.py"  
    work_dir = "/shared/anesthesia/wolosomething/delirium/cleanrun_01"  
    sweep_path = "/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/sweep"
    
    tune_grid = {'label': ['pseudo','only','full'],
                 'pl': [1,2,3],
                 'fr': [100,75,50,35,20],
                 'filter_keywords': [True, False],
                 'th': [90],
                 'lr': [2e-6],
                 'w_decay': [0.1], 
                 'n_batch': [16],
                 'lab_smooth': [0.0],
                 'cw': [0.05,0.25,0.5,0.75,0.95,1.0],
             }

    all_combinations = list(itertools.product(*tune_grid.values()))
    tune_grid = pd.DataFrame(all_combinations, columns=tune_grid.keys())

    tune_grid = tune_grid[~((tune_grid['label'] == 'only') & (tune_grid['fr'] != 100))]
    tune_grid = tune_grid[~((tune_grid['label'].isin(['only', 'full'])) & (tune_grid['pl'] != 1))]
    tune_grid = tune_grid[~((tune_grid['label'].isin(['pseudo', 'full'])) & (tune_grid['filter_keywords'] == False) & (tune_grid['fr'] != 100))]
    tune_grid = tune_grid[~((tune_grid['label'] == 'pseudo') & (tune_grid['cw'] > 0.5))]
    tune_grid = tune_grid[~((tune_grid['label'] == 'pseudo') & (tune_grid['filter_keywords'] == False) & (tune_grid['pl'] != 1))]
    tune_grid = tune_grid[~((tune_grid['label'].isin(['pseudo', 'full'])) & (tune_grid['filter_keywords'] == False) & (tune_grid['fr'] != 100))]
    tune_grid = tune_grid[~((tune_grid['label'].isin(['pseudo', 'full'])) & (tune_grid['pl'] != 1) & (tune_grid['fr'] != 100))]
    tune_grid = tune_grid[~((tune_grid['fr'] == 20))]

    tune_grid = tune_grid.sample(frac=1).reset_index(drop=True)
    
    for i in range(len(tune_grid)):

        out_dir = os.path.join(work_dir,'longformer','out','sweep','run_cw_final')

        label = tune_grid['label'][i]
        pl = tune_grid['pl'][i]
        fr = tune_grid['fr'][i]
        filter_keywords = tune_grid['filter_keywords'][i]
        th = tune_grid['th'][i]
        lr = tune_grid['lr'][i]
        w_decay = tune_grid['w_decay'][i]
        n_batch = tune_grid['n_batch'][i]
        lab_smooth = tune_grid['lab_smooth'][i]
        class_weighting = tune_grid['cw'][i]

        folder_name = ('fkw' + str(int(filter_keywords)) +
                       '_th' + str(th) +
                       '_fr' + str(fr) + 
                       '_pl' + str(pl) +
                       '_lr' + sigfigs(lr,1) +
                       '_wd' + sigfigs(w_decay,1) + 
                       '_nb' + str(n_batch) + 
                       '_ls' + sigfigs(lab_smooth,1) +
                       '_cw' + str(int(class_weighting * 100)) +
                       '_lab' + label)

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

        n_steps = int(n_notes * n_epoch / n_batch / n_grad)
        log_steps = int(n_steps * 0.05)

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
            "--pipeline", str(pl),
            "--seq_len", "4096",
            "--log_steps", str(log_steps),
            "--n_grad_accum", str(n_grad),
            "--n_grad_accum_eval", "1",
            "--n_batch", str(n_batch),
            "--n_batch_eval", "64",
            "--n_train_epochs", str(n_epoch),
            "--lr", str(lr),
            "--warmup_ratio", "0.05",
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
            "--wandb_pn", "sweep_run_cw_final",
            "--final_sweep", "False",
            "--early_stopping","True"
        ]

        print(f"\nRunning trial {folder_name}.")
        subprocess.run(command)

run_from_cp(n_epoch=3)
