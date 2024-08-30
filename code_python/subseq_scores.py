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
    AutoModelForSequenceClassification,
    AutoTokenizer, DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    LongformerForSequenceClassification,
    AutoConfig, TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForMaskedLM,
    TextClassificationPipeline,FillMaskPipeline,
    Trainer, HfArgumentParser,
    EarlyStoppingCallback, set_seed,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from transformers.pipelines.pt_utils import KeyDataset
import tokenizers
from peft import PeftConfig, PeftModel

import csv
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

def make_preds(sweep_dir,runs,datasets,return_scores=False,batch_size=16):
    y_test = {}
    for run in runs:
        mod = os.path.join(sweep_dir,'run_cw_' + sweep,run,'model_trainer')
        
        if not os.path.exists(mod):
            continue
        
        print(f"Predicting labels for {run}.", end='\r')
        conf = AutoConfig.from_pretrained(mod)
        model = LongformerForSequenceClassification.from_pretrained(mod,config=conf)
        tokenizer = AutoTokenizer.from_pretrained(mod)
        pl = TextClassificationPipeline(model=model,tokenizer=tokenizer,device=device,batch_size=batch_size,truncation=True)

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

def make_pl(run,sweep,sweep_dir,batch_size=16):
    mod = os.path.join(sweep_dir,'run_cw_' + sweep,run,'model_trainer')
    conf = AutoConfig.from_pretrained(mod)
    model = LongformerForSequenceClassification.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained(mod)
    pl = TextClassificationPipeline(model=model,tokenizer=tokenizer,device=device,batch_size=batch_size,truncation=True)
        
    return pl

def process_pred(pred):
    if pred['label'] == 'LABEL_1':
        score = pred['score']
    else:
        score = 1-pred['score']
    return(score)
        
def get_pred_trends(ds,pl,size=10,batch_size=16,shift=50):
    subseqs = {'id':[],'i':[],'text':[]}
    for o in ds:
        j = 0
        for i in range(0, len(o['text']) - size + 1,shift):
            subseqs['id'].append(o['id'])
            subseqs['i'].append(j)
            subseqs['text'].append(o['text'][i:i+size])
            j += 1
        if len(o['text']) % shift != 0:
            subseqs['id'].append(o['id'])
            subseqs['i'].append(j)
            subseqs['text'].append(o['text'][-size:])

    df = pd.DataFrame(subseqs)
    subseqs = Dataset.from_dict(subseqs)

    N = len(subseqs)
    counter = 1
    
    preds = []
    scores = []
    for r in pl(KeyDataset(subseqs,'text')):
        preds.append(r['label'])
        scores.append(r['score'])
        print(f"{counter}/{N}",end='\r')
        counter += 1
    
    df['preds'] = pd.Series(preds, name='preds')
    df['scores'] = pd.Series(scores, name='scores')
        
    return df

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(random.randint(1000, 9999))

os.environ['WANDB_DISABLED'] = 'true'

seed = 215
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
set_seed(seed)

sweep_dir = '/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/sweep'
data_dir = '/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/data'
out_dir = '/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/trends'
sweep = 'final'

datasets = {}
dat = read_data(os.path.join(data_dir,'tbl.csv.gz'),th=90,fr=100,exp='full')
datasets['train'] = Dataset.from_dict(dat['train'])

del dat

size = 250
shift = 50

#runs = ['fkw1_th70_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo',
#        'fkw1_th80_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo',
#        'fkw1_th90_fr100_pl1_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo']

runs = ['fkw1_th90_fr100_pl2_lr2.0e-06_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo']

for run in runs:
    pl = make_pl(run,sweep,sweep_dir,batch_size=64)
    
    for k,ds in datasets.items():
        print(f'Running predictions for {k}.')
        out_path = os.path.join(out_dir,run + '_' + k + '.csv.gz')
        trends = get_pred_trends(ds,pl,size=size,shift=shift,batch_size=64)
        trends.to_csv(out_path, index=False, compression='gzip')

        if device != 'cpu':
            gc.collect()
            torch.cuda.empty_cache()
            
llama_dir = '/shared/anesthesia/wolosomething/delirium/cleanrun_01/llama/out'
run = 'fkw1_th90_fr100_pl1_lr5.0e-05_wd1.0e-01_nb16_ls0.0e+00_cw5_labpseudo'
mod = os.path.join(llama_dir,run,'model_trainer')
conf = AutoConfig.from_pretrained(mod)
model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf).eval()
tokenizer = AutoTokenizer.from_pretrained(mod)

pl = TextClassificationPipeline(model=model,tokenizer=tokenizer,device=device,
                                batch_size=16,truncation=True,max_length=8192)

for k,ds in datasets.items():
    print(f'Running predictions for {k}.')
    out_path = os.path.join(out_dir,run + '_' + k + '.csv.gz')
    trends = get_pred_trends(ds,pl,size=size,shift=shift,batch_size=64)
    trends.to_csv(out_path, index=False, compression='gzip')

    if device != 'cpu':
        gc.collect()
        torch.cuda.empty_cache()
