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

        idn = row[idx_id]
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
            label = int(row[idx_label_icd])
            if train_on_expert:
                setn = 'train'                
        else:
            label = int(row[idx_label])
            
        if finetuning:
            if label != -1:
                d[setn]['text'].append(text) 
                d[setn]['id'].append(idn)
                d[setn]['labels'].append(label)
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

def group_preds(predictions,labels):
    
    N = len(labels)
    res_dict = dict()
    
    for i in range(N):
        idnum = labels[i]
        label = predictions[1][i]
        pred = predictions[0][i].argmax()
        val = predictions[0][i].max()
 
        try:
            res_dict[idnum]['pred'].append(pred)
            res_dict[idnum]['val'].append(val)
        except KeyError:
            res_dict[idnum] = {'label':label,'pred':[pred],'val':[val]}
            
    return(res_dict)


def majority_vote(res_dict):
    
    maj_dict = dict()

    for k,v in res_dict.items():
        
        if len(res_dict[k]['pred']) == 1:
            
            n_str = 1
            decision = res_dict[k]['pred'][0]
            
        else:
            
            n_str = len(res_dict[k]['pred'])
            decision_mag = [abs(inv_logit(v))-.5 for v in res_dict[k]['val']]
            decision = res_dict[k]['pred'][np.argmax(decision_mag)]
           
        if decision == res_dict[k]['label']:
            m = 1
        else:
            m = 0
                
        maj_dict[k] = [m,decision,res_dict[k]['label'],n_str]

    return(maj_dict)

def compute_eval_metrics(res,test_ids):
    
    grouped = group_preds(res,test_ids)
    mv = majority_vote(grouped)
    
    preds = [v[1] for k,v in mv.items()]
    labels = [v[2] for k,v in mv.items()]

    acc = evaluate.load('accuracy').compute(predictions=preds, references=labels)['accuracy']
    prec = evaluate.load('precision').compute(predictions=preds, references=labels)['precision']
    rec = evaluate.load('recall').compute(predictions=preds, references=labels)['recall']
    f1 = evaluate.load('f1').compute(predictions=preds, references=labels)['f1']
    
    return {'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec, 
        'batch_length': len(preds),'pred_positive': sum(preds), 'true_positive': sum(labels)}
