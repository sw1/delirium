import os
import pandas as pd
import math
import statsmodels.api as sm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth',75)

def merge_results(run,rf=2,stat='b_acc'):
    sweep_path = "/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/sweep"
    run = 'run_' + str(run)
    path = os.path.join(sweep_path,run)
    
    dfs = []
    for trial in os.listdir(path):
        fn = os.path.join(path,trial,"eval_res.csv")
        
        if os.path.exists(fn):
            df = pd.read_csv(fn)
            #df['cw'] = int(os.path.join(path,trial).split('/')[-1].split('_')[-1].replace('cw',''))/100
            df['path'] = os.path.join(path,trial).split('/')[-1]
            #df.to_csv(os.path.join(path,trial,"eval_res.csv"), index=False)
            df['score'] = math.sqrt(float(df['b_acc'].iloc[0]) / 0.5 / (abs(float(df['prop_diff'].iloc[0])) + 1))
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
        
def stat_summary(run,y):
    sweep_path = "/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/sweep"
    run = 'run_' + str(run)
    
    df = pd.read_csv(os.path.join(sweep_path,run + '.csv'))
    df['intercept'] = 1
    df['filter_keywords'] = df['filter_keywords'].astype(int)

    predictors = ['filter_keywords', 'th', 'lr', 'w_decay', 'eff_n_batch', 'lab_smooth', 'cw']
    df[predictors] = df[predictors] - df[predictors].mean()

    df['th_cw'] = df['th'] * df['cw']
    df['effnbatch_lr'] = df['eff_n_batch'] * df['lr']

    X = predictors + ['intercept', 'th_cw', 'effnbatch_lr']
    
    model = sm.OLS(df[y],df[X])
    results = model.fit()
    print(results.summary())
    
merge_results('cw_1',rf=3,stat='b_acc')
#stat_summary('cw_0',y='b_acc')
#stat_summary('cw_0',y='score')
