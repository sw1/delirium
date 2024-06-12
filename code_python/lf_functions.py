import os
import sys
import shutil

import gzip
import csv
import random
import numpy as np
import pandas as pd
import math
import statistics
import pickle 
import re

import matplotlib.pyplot as plt
from pynvml import *
from plotnine import *

import torch
from torch import nn

import accelerate
import transformers
from transformers import (
    AutoTokenizer, pipeline, DataCollatorForLanguageModeling, 
    AutoModelForSequenceClassification, AdamW, AutoModelForMaskedLM, 
    AutoConfig, TrainingArguments, Trainer, TextClassificationPipeline,
    DataCollatorForLanguageModeling, FillMaskPipeline, LongformerModel,
    LongformerTokenizer, LongformerForMaskedLM,
    EarlyStoppingCallback, IntervalStrategy,
)
import tokenizers
from tokenizers import (
    decoders, models, normalizers, pre_tokenizers, processors,
    trainers,Tokenizer,AddedToken,
)
from transformers.integrations import *
import evaluate

import datasets
from datasets import (
    load_dataset, Dataset, load_metric, DatasetDict,
    concatenate_datasets, interleave_datasets,
)

from sklearn.utils import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def read_data(fn,st,train_on_expert=True,finetuning=True):
    d = {s: {'id':[],'text':[],'labels':[]} 
         for s in ['train','val','test_icd','test_expert','train_expert','heldout_expert']}
    
    reader = csv.reader(gzip.open(fn,mode='rt',encoding='utf-8'))

    header = next(reader, None)

    idx_id = header.index('id')
    idx_text = header.index('hpi_hc')
    idx_label = header.index('label')
    idx_label_icd = header.index('label_icd')
    idx_set = header.index('set')
    
    if st:
        idx_label_pseudo = header.index('label_pseudo')
    
    for row in reader:

        try:
            idn = int(row[idx_id])
        except ValueError:
            idn = int(float(row[idx_id]))

        setn = row[idx_set]        
        text = row[idx_text]
        
        if setn == 'train' or setn == 'val':
            if st:
                label = int(row[idx_label_pseudo])
            else:
                label = int(row[idx_label_icd])
        elif setn == 'test_icd':
            label = int(row[idx_label_icd])
        elif setn == 'train_expert':
            label = int(row[idx_label])
            if train_on_expert:
                setn = 'train'                
        else:
            label = int(row[idx_label])
            
        if finetuning:
            if label != -1:
                d[setn]['text'].append(text) 
                d[setn]['id'].append(idn)
                d[setn]['labels'].append(label)
                
                # add heldout samples to icd test set w/ icd labels
                if setn == 'heldout_expert':
                    label_icd = int(row[idx_label_icd])
                    if label_icd != -1:                    
                        d['test_icd']['text'].append(text) 
                        d['test_icd']['id'].append(idn)
                        d['test_icd']['labels'].append(label_icd)
        else:
            if setn == 'train' or setn == 'val':
                d[setn]['labels'].append(label)
                d[setn]['text'].append(text) 
                d[setn]['id'].append(idn)
            else:
                if label != -1:
                    d[setn]['text'].append(text) 
                    d[setn]['id'].append(idn)
                    d[setn]['labels'].append(label)
                    
    return(d)

def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

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

def compute_eval_metrics(res,test_ids,method=1):
    
    grouped = group_preds(res,test_ids)
    mv = majority_vote(grouped,method)
    
    preds = list()
    labels = list()
    scores = list()
    for k,v in mv.items():  
        preds.append(int(v[1]))
        labels.append(v[2]) 
        scores.append(v[4])

    auc = evaluate.load('roc_auc').compute(references=labels, prediction_scores=scores)['roc_auc']
    acc = evaluate.load('accuracy').compute(predictions=preds, references=labels)['accuracy']
    prec = evaluate.load('precision').compute(predictions=preds, references=labels)['precision']
    rec = evaluate.load('recall').compute(predictions=preds, references=labels)['recall']
    f1 = evaluate.load('f1').compute(predictions=preds, references=labels)['f1']
    
    return {'accuracy': acc, 'f1': f1, 'auc': auc,'precision': prec, 'recall': rec, 
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

def balance_data(x,s=123,cores=1):
    positive_label = x.filter(lambda example: example['labels']==1, num_proc=cores) 
    negative_label = x.filter(lambda example: example['labels']==0, num_proc=cores)

    n_us = n_upsamp(positive_label,negative_label)

    random.seed(s)
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
