import os
import sys
import gc
import subprocess

import random
import numpy as np
from scipy.special import softmax
import pandas as pd
import math
import pickle 
import re

import matplotlib.pyplot as plt
from pynvml import *
from plotnine import *

import logging
import wandb

import tarfile

import torch
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoTokenizer, DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    LongformerForSequenceClassification,
    AutoConfig, TrainingArguments,
    AutoModelForMaskedLM,
    TextClassificationPipeline,FillMaskPipeline,
    Trainer, HfArgumentParser,
    EarlyStoppingCallback, set_seed,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from transformers.pipelines.pt_utils import KeyDataset
import tokenizers

import evaluate
from datasets import load_dataset, Dataset,load_from_disk

from sklearn.utils import compute_class_weight
from sklearn.metrics import (precision_recall_fscore_support,
                             confusion_matrix,
                             balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split

from typing import Optional
from dataclasses import dataclass, field

# import custom functions
from lf_functions import (
    read_data, softmax, compute_eval_metrics,
    balance_data, get_class_weights, check_and_save_params,
    pull_results, print_vars, process_log_history, final_preds,
)

def compute_metrics(labels,preds,scores):
    auc = evaluate.load('roc_auc').compute(references=labels, prediction_scores=scores)['roc_auc']
    acc = evaluate.load('accuracy').compute(predictions=preds, references=labels)['accuracy']
    prec = evaluate.load('precision').compute(predictions=preds, references=labels)['precision']
    rec = evaluate.load('recall').compute(predictions=preds, references=labels)['recall']
    f1 = evaluate.load('f1').compute(predictions=preds,references=labels)['f1']
    bacc = balanced_accuracy_score(y_true=labels,y_pred=preds)
    ppp = sum(preds)/len(preds)
    ptp = sum(labels)/len(labels)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp)
    
    return {
        'accuracy': acc,
        'b_accuracy': bacc,
        'f1': f1,
        'auc': auc,
        'precision': prec,
        'recall': rec,
        'spec': specificity,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'prop_pred_positive': ppp,
        'prop_true_positive': ptp
    }

def get_params(string):
    parts = string.split('_')
    result = {}
    for part in parts:
        if part.startswith('lab'):
            result['lab'] = part[3:]
        else:
            for i, char in enumerate(part):
                if char.isdigit() or char == '.':
                    key = part[:i]
                    value = part[i:]
                    result[key] = value
                    break
    
    return(result)

def make_preds(sweep_dir,runs,datasets,return_scores=False):
    y_test = {}
    for run in runs:
        mod = os.path.join(sweep_dir,'run_cw_' + sweep,run,'model_trainer')
        
        if not os.path.exists(mod):
            continue
        
        print(f"Predicting labels for {run}.", end='\r')
        conf = AutoConfig.from_pretrained(mod)
        model = LongformerForSequenceClassification.from_pretrained(mod,config=conf)
        tokenizer = AutoTokenizer.from_pretrained(mod)
        pl = TextClassificationPipeline(model=model,tokenizer=tokenizer,device=device,batch_size=16,truncation=True)

        y_test[run] = {}
        y_test[run]['params'] = get_params(run)
        y_test[run]['results'] = {}
        for k,v in datasets.items():            
            y_hats = []
            y_scores = []
            for pred in pl(KeyDataset(v,'text')):
                if pred['label'] == 'LABEL_1':
                    y_hats.append(1)
                    y_scores.append(pred['score'])
                else:
                    y_hats.append(0)
                    y_scores.append(1-pred['score'])

            y_test[run]['results'][k] = compute_metrics(v['labels'],y_hats,y_scores)
            
            if return_scores:
                y_test[run]['results'][k]['scores'] = y_scores

    return y_test

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth',75)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(random.randint(1000, 9999))

os.environ['WANDB_DISABLED'] = 'true'

sweep_dir = '/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/sweep'
data_dir = '/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/data'
sweep = 'final'

runs = os.listdir(os.path.join(sweep_dir, 'run_cw_' + sweep))

current_file = os.path.join(sweep_dir,'run_cw_' + sweep + '_results.csv.gz')
if os.path.exists(current_file):
    current_res = pd.read_csv(os.path.join(sweep_dir,'run_cw_' + sweep + '_results.csv.gz'))
    runs_predicted = []
    for i,row in current_res.iterrows():
        runs_predicted.append('fkw' + str(row['fkw']) + '_th' + str(row['th']) + '_fr' + str(row['fr']) + '_pl' + str(row['pl']) + '_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00' + '_cw' + str(row['cw']) + '_lab' + row['lab'])    
    runs = set(runs).symmetric_difference(runs_predicted)
    print(f"Adding {len(runs)} new runs.")
    
dat = read_data(os.path.join(data_dir,'tbl.csv.gz'),th=90,fr=100,exp='pseudo')

dat['heldout_expert_filtered'] = dat['heldout_expert'] 
for i in range(len(dat['heldout_expert_filtered']['text'])):
    note = dat['heldout_expert_filtered']['text'][i]
    note = note.replace('delirium','').replace('encephalopathy','')
    dat['heldout_expert_filtered']['text'][i] = note

datasets = {}
datasets['val'] = Dataset.from_dict(dat['val'])
datasets['icd'] = Dataset.from_dict(dat['heldout_icd'])
datasets['expert'] = Dataset.from_dict(dat['heldout_expert'])
datasets['filtered'] = Dataset.from_dict(dat['heldout_expert_filtered'])

y_test = make_preds(sweep_dir,runs,datasets)

sets = ['icd','expert','filtered']
i = 0
df = pd.DataFrame()
for v in y_test.values():
    for s in sets:
        d = {**v['params'],**v['results'][s]}
        d['set'] = s
        set_df = pd.DataFrame(d,index=[0])
        i += 1
        df = pd.concat([df, set_df], ignore_index=True)
    
columns = ['lab','set'] + [col for col in df.columns if col not in ['lab','set']]
df = df[columns]
df = df.sort_values(by='b_accuracy',ascending=False)
print(f"\n{len(df)} additional runs to update.\n")

if os.path.exists(current_file):
    df = pd.concat([current_res,df],ignore_index=True)
print(f"\n{len(df)} current runs after update.\n")
    
df.to_csv(os.path.join(sweep_dir,'run_cw_' + sweep + '_results.csv.gz'), index=False, compression='gzip')

print(df)
