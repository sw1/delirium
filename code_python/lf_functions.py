import csv
import gzip
import shutil
import time
import json
import os
import numpy as np
import pandas as pd
import pickle
import torch
import evaluate
import random
from datasets import interleave_datasets, concatenate_datasets
from scipy.special import softmax
import matplotlib.pyplot as plt
from pynvml import *
from plotnine import *

from sklearn.metrics import (precision_recall_fscore_support,
                             balanced_accuracy_score,
)

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def read_data(fn,exp,th=None,fr=None):
    d = {s: {'id':[],'text':[],'labels':[]} 
         for s in ['train','val','heldout_icd','heldout_expert']}

    reader = csv.reader(gzip.open(fn,mode='rt',encoding='utf-8'))

    header = next(reader, None)
    
    idx_id = header.index('id')
    idx_text = header.index('hpi_hc')
    idx_hpi = header.index('hpi')
    idx_hc = header.index('hc')
    idx_label_expert = header.index('label')
    idx_label_icd = header.index('label_icd')
    idx_set = header.index('set')

    if exp == 'full':
        idx_label = header.index('label_fullexpert')
    elif exp == 'only':
        idx_label = header.index('label')
    elif exp == 'pseudo':
        idx_label = header.index('label_pseudo_th' + th + '_fr' + fr)
    elif exp == 'icd':
        idx_label = header.index('label_icd')

    for row in reader:
        label = -1
        label_icd = -1
        
        try:
            idn = int(row[idx_id])
        except ValueError:
            idn = int(float(row[idx_id]))

        setn = row[idx_set]
        
        if exp == 'pretrain':
            d[setn]['id'].append(idn)
            d[setn]['text'].append(row[idx_text]) 
            d[setn]['labels'].append(-1)
        else:
            if setn == 'train':
                label = int(row[idx_label])
            elif setn == 'val' and not exp == 'icd':
                label = int(row[idx_label_expert])
            elif setn == 'val' and exp == 'icd':
                label = int(row[idx_label])
            elif setn == 'heldout_expert':
                label = int(row[idx_label_expert])
                label_icd = int(row[idx_label_icd])
            else:
                break
            
            if label != -1:
                d[setn]['id'].append(idn)
                d[setn]['text'].append(row[idx_text]) 
                d[setn]['labels'].append(label)

            if setn == 'heldout_expert' and label_icd != -1:
                d['heldout_icd']['id'].append(idn)
                d['heldout_icd']['text'].append(row[idx_text])
                d['heldout_icd']['labels'].append(label_icd)      
                
    return(d)

def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def group_preds(predictions,ids):
    
    N = len(ids)
    res_dict = dict()
    
    for i in range(N):
        idnum = ids[i]
        label = predictions[1][i]
        val = softmax(predictions[0][i])[-1]
        pred = np.where(val > 0.5,1,0)
 
        try:
            res_dict[idnum]['pred'].append(pred)
            res_dict[idnum]['val'].append(val)
        except KeyError:
            res_dict[idnum] = {'label':label,'pred':[pred],'val':[val]}
            
    return(res_dict)

def majority_vote(res_dict,method):
    
    maj_dict = dict()

    for k,v in res_dict.items():
        
        if len(v['pred']) == 1:
            
            n_str = 1
            decision = v['pred'][0]
            score = v['val'][0]
            
        else:
        
            n_str = len(v['pred'])
            
            if method == 1:
        
                score_idx = abs(np.array(v['val'])-.5).argmax()
                score = v['val'][score_idx]
                decision = v['pred'][score_idx]
                
            elif method == 2:
                
                score = np.mean(v['val'])
                
                if score > 0.5:
                    decision = 1
                else:
                    decision = 0
            else:
                print("Method must be 1 for majority vote or 2 for average.")
                return  
           
        if decision == v['label']:
            m = 1
        else:
            m = 0
                
        maj_dict[k] = [m,decision,v['label'],n_str,score]

    return(maj_dict)

def compute_eval_metrics(res,test_ids):
    
    scores = softmax(res,axis=1)
    preds = np.argmax(scores, axis=1)

    auc = evaluate.load('roc_auc').compute(references=labels, prediction_scores=scores)['roc_auc']
    acc = evaluate.load('accuracy').compute(predictions=preds, references=labels)['accuracy']
    prec = evaluate.load('precision').compute(predictions=preds, references=labels)['precision']
    rec = evaluate.load('recall').compute(predictions=preds, references=labels)['recall']
    f1 = evaluate.load('f1').compute(predictions=preds, references=labels)['f1']
    bacc = balanced_accuracy_score(y_true=labels,y_pred=preds)
    
    return {'accuracy': acc, 'b_accuracy': bacc,
            'f1': f1, 'auc': auc,'precision': prec, 'recall': rec, 
        'batch_length': len(preds),'pred_positive': sum(preds), 'true_positive': sum(labels)}

def n_upsamp(positive_label,negative_label):
    len_pos = len(positive_label)
    len_neg = len(negative_label)
    
    if (len_pos > len_neg):
        n_upsamp = len(positive_label) // len(negative_label)
    elif(len_neg > len_pos):
        n_upsamp = len(negative_label) // len(positive_label)
    else:
        n_upsamp = 0
        
    return(n_upsamp)
    
def balance_data(x,cores=1):
    positive_label = x.filter(lambda example: example['labels']==1, num_proc=cores) 
    negative_label = x.filter(lambda example: example['labels']==0, num_proc=cores)

    n_us = n_upsamp(positive_label,negative_label)

    seeds = random.sample(range(9999), n_us)

    balanced_data = None
    for ss in seeds:
        if balanced_data:
            balanced_data = concatenate_datasets([balanced_data, interleave_datasets([
                positive_label.shuffle(seed=ss), 
                negative_label.shuffle(seed=ss)
            ])])
        else:
            balanced_data = interleave_datasets([positive_label, negative_label])

    return(balanced_data)

def get_class_weights(data, num_labels):
    #class_weights = (1/pd.DataFrame(data).labels.value_counts(normalize=True).sort_index()).tolist()
    class_weights = (pd.DataFrame(data).shape[0]/(pd.DataFrame(data).labels.value_counts(normalize=False) * num_labels).sort_index()).tolist()
    class_weights = torch.as_tensor(class_weights)
    class_weights = class_weights/class_weights.sum()
    
    return(class_weights)

def print_vars(args,file=None):
    for key, value in args.items():
        print(f"{key}: {value}",file=file)
        
def check_and_save_params(model_args, tune_args, sweep_args, out_dir, folder_fn):

    params = {**vars(model_args), **vars(tune_args), **vars(sweep_args)}
    param_path = os.path.join(out_dir,'params.json')
    folder_fn_old = folder_fn
    
    ft_dir = os.path.dirname(os.path.dirname(out_dir))
    
    if os.path.isdir(ft_dir):
        folder_names = [d for d in os.listdir(ft_dir) if folder_fn_old in d] 
        match = False
        empty = False
        if len(folder_names) > 0:
            params_new = json.loads(json.dumps(params))
            for fn in folder_names:
                param_path_old = os.path.join(ft_dir,fn,'final_model_finetune','params.json')
                
                if os.path.exists(param_path_old):
                    with open(param_path_old,'r') as f:
                        params_old = json.load(f)
                else:
                    empty = True
                    
                if empty or params_old == params:
                    folder_fn = fn
                    
                    if tune_args.override_prompt:
                        while True:
                            user_input = input(f"\nSame model parameterization exists for run {folder_fn}. How would you like to proceed? (overwrite/rename/exit):   ")
                            if user_input.lower() in ['overwrite','o']:
                                print(f"Overwriting {folder_fn}.")
                                out_dir = os.path.dirname(out_dir)
                                param_path = os.path.join(out_dir,'params.json')
                                shutil.rmtree(out_dir)
                                match = True
                                break
                            elif user_input.lower() in ['rename','r']:
                                folder_fn += '_' + str(round(time.time()))
                                print(f"Changing folder name to {folder_fn}")
                                out_dir = out_dir.replace(folder_fn_old,folder_fn)
                                param_path = os.path.join(out_dir,'params.json')
                                match = True
                                break
                            elif user_input.lower() in ['exit','e']:
                                print('Ending run.')
                                sys.exit()
                            else:
                                print("Invalid input. Please enter 'overwrite', 'rename', or 'exit'.")
                    else:
                        print(f"\nSame model parameterization exists for run {folder_fn}. Overwriting folder.")
                        out_dir = os.path.dirname(out_dir)
                        param_path = os.path.join(out_dir,'params.json')
                        shutil.rmtree(out_dir)
                        match = True
                        break
                    
            if not match:
                folder_fn += '_' + str(round(time.time()))
                print(f"\nSimilar model with different parameterization exists, changing folder name to {folder_fn}.")
                out_dir = out_dir.replace(folder_fn_old,folder_fn)
                param_path = os.path.join(out_dir,'params.json')

    os.makedirs(out_dir)    
    with open(param_path, 'w') as f:
        json.dump(params, f)
            
    return out_dir, folder_fn


def pull_results(log_history):
    res_eval = []
    res_loss = []
    res_lr = []
    e_step = None
    t_step = None
    for r in log_history:
        try:
            res_eval.append([r['step'],r['eval_loss'],
                             r['eval_b_accuracy'],r['eval_f1']])
            e_loss = r['eval_loss']
            e_step = r['step']
            if e_step != None and t_step != None:
                if e_step == t_step:
                    res_loss.append([e_step,e_loss,t_loss])
        except KeyError:
            try:
                t_loss = r['loss']
                t_step = r['step']
                res_lr.append([t_step,r['learning_rate']])
                if e_step != None and t_step != None:
                    if e_step == t_step:
                        res_loss.append([e_step,e_loss])
            except KeyError:
                next
    
    return res_eval, res_loss, res_lr

def process_log_history(sweep_args,log_history,out_dir):

    # save trainer_history
    with open(os.path.join(out_dir,'log_history.pkl'), 'wb') as f:
        pickle.dump(log_history, f)
        
    res_eval, res_loss, res_lr = pull_results(log_history)

    # loss figure
    df = pd.DataFrame(res_loss,columns=['step','eval_loss','train_loss'])
    plt.figure()
    plt.plot(df['step'],df['eval_loss'],label='eval_loss')
    plt.plot(df['step'],df['train_loss'],label='train_loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir,'figure1.png'))
    
    # lr figure
    df = pd.DataFrame(res_lr,columns=['step','lr'])
    plt.figure()
    plt.plot(df['step'],df['lr'])
    plt.legend()
    plt.savefig(os.path.join(out_dir,'figure2.png'))

    if sweep_args.train_method == 'finetune':
        # performance figure
        df = pd.DataFrame(res_eval,columns=['step','loss','b_acc','f1'])
        plt.figure()
        plt.plot(df['step'],df['b_acc'],label='b_acc')
        plt.plot(df['step'],df['f1'],label='f1')
        plt.legend()
        plt.savefig(os.path.join(out_dir,'figure3.png'))
        
def final_preds(out_dir,args=None,*kwargs):
    print('\Final predictions.')

    test_results = {}

    with open(os.path.join(out_dir,'test_results.dat'), 'w') as f:
        if args is not None:
            print('\nRun args:')
            print_vars(args,file=f)
            
        for y_name,y_x in kwargs.items():
            y_hat = trainer.predict(y_x)
            test_results[y_name] = y_hat

            print(f"\nResults for table {y_name}",file=f)
            print("\nResults for table {y_name}.")
            for k,v in y_hat[2].items():
                print(f"{k}: {v}",file=f)
                print(f"{k}: {v}")
            print('\n',file=f)

    with open(os.path.join(out_dir,'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results, f)
