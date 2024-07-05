import os
import pandas as pd
import re
import sys
import shutil
import itertools

pd.set_option('display.max_rows', 500)

run = 0

sweep_path = "/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/sweep"

def fix_ebatch(run,sweep_path):
    run = 'run_' + str(run)
    path = os.path.join(sweep_path,run)

    dfs = []
    for trial in os.listdir(path):
        if trial.find('nb64') != -1:
            fn = os.path.join(path,trial,"eval_res.csv")
            if os.path.exists(fn):
                df = pd.read_csv(fn)
                df.loc[0, 'eff_n_batch'] = 64
                df.to_csv(fn,index=False)
                
def check_completion(run,sweep_path):
    run = 'run_' + str(run)
    path = os.path.join(sweep_path,run)
    
    tune_grid = {'filter_keywords': [False,True],
             'lr': [2e-6, 8e-6, 2e-5],
             'w_decay': [0.001, 0.01, 0.1, 1.0],
             'n_batch': [8, 16, 32, 64],
             'lab_smooth': [0.0, 1e-1, 3e-1],
             }
    all_combinations = list(itertools.product(*tune_grid.values()))
    tune_grid = pd.DataFrame(all_combinations, columns=tune_grid.keys())
    print(f"\nProjected number of trials: {len(tune_grid)}.\n")
    
    dfs = []
    folders = os.listdir(path)
    print(f"\nCurrent number of completed trials: {len(folders)}.\n\n")
    for trial in folders:
        fn = os.path.join(path,trial,"eval_res.csv")
        
        if not os.path.exists(fn):
            print(f"Missing trial: {fn}.\n")
    
    
#fix_ebatch(run,sweep_path)
check_completion(run,sweep_path)
