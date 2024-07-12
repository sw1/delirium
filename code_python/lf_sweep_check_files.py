import os
import pandas as pd
import re
import sys
import shutil
import itertools
from lf_functions import sigfigs

run = 'cw_0'

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
                
def check_completion(run,sweep_path,rm=False):
    run = 'run_' + run
    path = os.path.join(sweep_path,run)
    
    tune_grid = {'filter_keywords': [False],
                 #'filter_keywords': [True,False],
                 #'th': [70,80,90],
                 'th': [80,90],
                 'lr': [2e-6,8e-6],
                 #'lr': [2e-6,8e-6,2e-5],
                 #'w_decay': [0.1,0.01,.001],
                 'w_decay': [0.01],
                 #'n_batch': [8,16,32,64],
                 'n_batch': [8,16,32],
                 #'lab_smooth': [0.3,0.15,0.0],
                 'lab_smooth': [0.0],
                 #'class_weighting': [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1.0],
                 'class_weighting': [0.05,0.1,0.25,0.75],
                 }

    all_combinations = list(itertools.product(*tune_grid.values()))
    tune_grid = pd.DataFrame(all_combinations, columns=tune_grid.keys())
    print(f"\nProjected number of target trials: {len(tune_grid)}.\n")
    
    dfs = []
    folders = os.listdir(path)
    counter = 0
    print(f"\nCurrent number of completed trials: {len(folders)}.\n")
    print('\nIncomplete trials:')
    for trial in folders:
        if not os.path.exists(os.path.join(path,trial,"eval_res.csv")) and trial.find('fkw') != -1:
            print(trial)
            counter += 1
            if rm:
                shutil.rmtree(os.path.join(path,trial))
    print(f"N={counter}\n")
    
    counter = 0
    print('\nRemaining trials:')
    for i in range(len(tune_grid)):

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

        if not os.path.exists(os.path.join(path,folder_name,"eval_res.csv")) and folder_name.find('fkw') != -1:
            print(folder_name)
            counter += 1
    print(f"N={counter}\n")
    
    
#fix_ebatch(run,sweep_path)
check_completion(run,sweep_path,rm=False)
