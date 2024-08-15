import os
import pandas as pd
import math
import statsmodels.api as sm
import argparse
import shutil

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth',75)

def stat_summary(run,y='b_acc'):
    sweep_path = "/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/sweep"
    run = 'run_cw_' + str(run)
    
    df = pd.read_csv(os.path.join(sweep_path,run + '.csv'))
    df['intercept'] = 1
    df['filter_keywords'] = df['filter_keywords'].astype(int)

    predictors = ['filter_keywords', 'th', 'lr', 'w_decay', 'eff_n_batch', 'cw']
    df[predictors] = df[predictors] - df[predictors].mean()

    df['th_cw'] = df['th'] * df['cw']
    df['effnbatch_lr'] = df['eff_n_batch'] * df['lr']

    X = predictors + ['intercept', 'th_cw', 'effnbatch_lr']
    
    model = sm.OLS(df[y],df[X])
    results = model.fit()
    print(results.summary())
    
def merge_results(run,rf=1,stat='b_acc',stat_summary=False):
    sweep_path = "/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/sweep"
    run = 'run_cw_' + str(run)
    path = os.path.join(sweep_path,run)
    
    dfs = []
    for trial in os.listdir(path):
        fn = os.path.join(path,trial,"eval_res.csv")
        
        if os.path.exists(fn):
            df = pd.read_csv(fn)
            #df['cw'] = int(os.path.join(path,trial).split('/')[-1].split('_')[-1].replace('cw',''))/100
            df['path'] = os.path.join(path,trial).split('/')[-1]
            #df.to_csv(os.path.join(path,trial,"eval_res.csv"), index=False)
            #df['score'] = math.sqrt(float(df['b_acc'].iloc[0]) / 0.5 / (abs(float(df['prop_diff'].iloc[0])) + 1))
            dfs.append(df)
        else:
            continue

    df = pd.concat(dfs, ignore_index=True).sort_values(by=stat,ascending=False)    
    df.to_csv(os.path.join(sweep_path,run + '.csv'), index=False)

    if rf is not None:
        top_n = len(df) // rf
        print(df.iloc[:top_n])
        print(f"\n{top_n}/{len(df)} total trials.\n\n")
        if len(df) > top_n:
            print(df.iloc[top_n:])
            print(f"\n{len(df)} total trials.\n")
    else:
        print(df)
        print(f"\n{len(df)} total trials.\n")
        
def clean_results(run, rm=False):
    sweep_path = "/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/sweep"
    run = 'run_cw_' + str(run)
    path = os.path.join(sweep_path,run)
    
    n_incomplete = 0
    n_complete = 0
    for trial in os.listdir(path):
        if not os.path.exists(os.path.join(path,trial,"eval_res.csv")):
            if rm:
                shutil.rmtree(os.path.join(path,trial))
            n_incomplete += 1
        else:
            n_complete += 1
            
    print(f"{n_complete} complete runs | {n_incomplete} incomplete runs.\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, required=True)
    parser.add_argument('--rf', type=int, required=False, default=1)
    parser.add_argument('--stat', type=str, required=False, default='b_acc')
    parser.add_argument('--stat_summary', type=bool, required=False, default=False)
    parser.add_argument('--y', type=str, required=False, default='b_acc')
    parser.add_argument('--rm', type=bool, required=False, default=False)
    args = parser.parse_args()

    merge_results(args.run, args.rf, args.stat, args.stat_summary)
    stat_summary(args.run)
    clean_results(args.run,args.rm)
