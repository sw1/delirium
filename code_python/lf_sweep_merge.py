import os
import pandas as pd

pd.set_option('display.max_rows', 500)

def merge_results(run):
    sweep_path = "/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/sweep"
    run = 'run_' + str(run)
    path = os.path.join(sweep_path,run)
    
    dfs = []
    for trial in os.listdir(path):
        fn = os.path.join(path,trial,"eval_res.csv")
        
        if os.path.exists(fn):
            df = pd.read_csv(fn)
            df['path'] = os.path.join(path,trial)
            dfs.append(df)
        else:
            next

    df = pd.concat(dfs, ignore_index=True).sort_values(by='b_acc',ascending=False)    
    df.to_csv(os.path.join(sweep_path,run,'.csv'), index=False)
    
    print(f"\n{len(df)} total trials.\n")
    print(df)
    
merge_results(0)
