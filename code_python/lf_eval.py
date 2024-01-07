import os
import sys

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

def read_data(fn):
    d = {s: {'id':[],'text':[], 'hpi':[], 'labels':[], 'icd_sum':[], 'labels_h':[]} 
         for s in ['train','val','test_icd','test_haobo']}
    
    reader = csv.reader(gzip.open(fn,mode='rt',encoding='utf-8'))

    header = next(reader, None)

    idx_id = header.index('id')
    idx_hpi = header.index('hpi')
    idx_label = header.index('label_icd')
    idx_text = header.index('hpi_hc')
    idx_label_h = header.index('label')
    idx_set = header.index('set')
    idx_icd_sum = header.index('icd_sum')

    for row in reader:

        try:
            idn = int(row[idx_id])
        except ValueError:
            idn = int(float(row[idx_id]))

        idn = row[idx_id]
        setn = row[idx_set]        
        text = row[idx_text]
        hpi = row[idx_hpi]
        icd_sum = int(row[idx_icd_sum])
        label = int(row[idx_label])
        
        label_h = row[idx_label_h]
        if label_h == 'NA':
            label_h = -1
        else:
            label_h = int(label_h)

        d[setn]['labels'].append(label)
        d[setn]['labels_h'].append(label_h)
        d[setn]['text'].append(text) 
        d[setn]['id'].append(idn)
        d[setn]['hpi'].append(hpi)
        d[setn]['icd_sum'].append(icd_sum)

    return(d)

def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

print(torch.cuda.device_count())

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(random.randint(1000, 9999))

pipelines = ('finetune repo model',
             'finetune pretrained model',
             'finetune pretrained model that used custom tokenizer')

s1 = 1234 
seq_len = 4096
balance_data = True
no_punc = False
subchapter = False

pipeline = int(sys.argv[1]) 
update_labels = int(sys.argv[2])

work_dir = '/home/swolosz1/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer'
data_dir = os.path.join(work_dir,'data')
out_dir = os.path.join(work_dir,'out')

if no_punc:
    tbl_fn = 'tbl_to_python_231205.csv.gz'
    tbl_tree_fn = 'tbl_to_python_231205_count_del.csv.gz'
    print('Fitting model without punctuation.')
else:
    tbl_fn = 'tbl_to_python_231229.csv.gz'
    tbl_tree_fn = 'tbl_to_python_231229_count_del.csv.gz'
    out_dir = os.path.join(out_dir,'punc')
    print('Fitting model with punctuation.')

token_dir = os.path.join(out_dir,'token')
pretrain_dir = os.path.join(out_dir,'pretrain')
finetune_dir = os.path.join(out_dir,'finetune')
sweep_dir = os.path.join(out_dir,'sweep')

model_pretrain = os.path.join(pretrain_dir,'model_pretrain')
model_token_pretrain = os.path.join(pretrain_dir,'model_token_pretrain')

out_token = os.path.join(token_dir,'custom_tokenizer.json')

if update_labels == 1:
    if subchapter:
        finetune_dir = os.path.join(finetune_dir,'updated_labels_subchapter')
        tbl_tree_fn = re.sub('count_del.csv.gz','count_del_subchapter.csv.gz',tbl_tree_fn)
        print('Updating to Subchapter tree labels.')
    else:
        finetune_dir = os.path.join(finetune_dir,'updated_labels')
        print('Updating to ICD tree labels.')
    dat = read_data(os.path.join(data_dir,tbl_tree_fn))
if update_labels == 0:
    dat = read_data(os.path.join(data_dir,tbl_fn))
    
out_finetune = os.path.join(finetune_dir,'final_model_finetune')
out_pretrain_finetune = os.path.join(finetune_dir,'final_model_pretrain_finetune')
out_token_pretrain_finetune = os.path.join(finetune_dir,'final_model_token_pretrain_finetune')

if pipeline == 1: # finetune with repo model
    mod = os.path.join(out_finetune,'model')
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                              use_fast=True,max_length=seq_len)
elif pipeline == 2: # finetune with pretrained model
    mod = os.path.join(out_pretrain_finetune,'model')
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                              use_fast=True,max_length=seq_len)
elif pipeline == 3: # finetune with pretrained model that used custom tokenizer
    mod = os.path.join(out_token_pretrain_finetune,'model')
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained(token_dir,use_fast=True,max_length=seq_len)
    
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

conf.hidden_dropout_prob=0.1
#conf.attention_probs_dropout_prob=0.1
conf.classifier_dropout=0.1

def tokenize_dataset(data):
    
    data["text"] = [
        line for line in data["text"] if len(line) > 0 and not line.isspace()
    ]
        
    return tokenizer(
        data['text'],
        padding='max_length', 
        truncation=True,
        max_length=seq_len,
        return_special_tokens_mask=True,
    )

d_val = Dataset.from_dict(dat['val'])
d_test_haobo = Dataset.from_dict(dat['test_haobo'])
d_test_icd = Dataset.from_dict(dat['test_icd'])

print('Tokenizing validation data.')
d_val = d_val.map(tokenize_dataset,batched=True,num_proc=16)
print('Tokenizing testing data.')
d_test_haoboset_hpihc = d_test_haobo.map(tokenize_dataset,batched=True,num_proc=16)
d_test_icdset_hpihc = d_test_icd.map(tokenize_dataset,batched=True,num_proc=16)
d_test_haobolabels_hpihc = d_test_haobo.remove_columns('labels').rename_column('labels_h','labels').map(tokenize_dataset,batched=True,num_proc=16)

def compute_metrics2(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    preds_scores = np.max(logits, axis=-1)
    
    auc = evaluate.load("roc_auc").compute(references=labels, prediction_scores=preds_scores)['roc_auc']
    acc = evaluate.load('accuracy').compute(predictions=preds, references=labels)['accuracy']
    prec = evaluate.load("precision").compute(predictions=preds, references=labels)["precision"]
    rec = evaluate.load("recall").compute(predictions=preds, references=labels)["recall"]
    f1 = evaluate.load("f1").compute(predictions=preds, references=labels)['f1']

    return {"accuracy": acc, "f1": f1, "auc": auc, "precision": prec, "recall": rec, 
            "batch_length": len(preds),"pred_positive": sum(preds), "true_positive": sum(labels)}

print('Validation groups.')
print([d_val['labels'].count(0),d_val['labels'].count(1)])

print('Testing ICD groups.')
print([d_test_icd['labels'].count(0),d_test_icd['labels'].count(1)])

print('Testing Haobo groups.')
print([d_test_haobo['labels'].count(0),d_test_haobo['labels'].count(1)])

class cTrainer(Trainer):
    pass
    
trainer = cTrainer(
    model=model.to(device),
    eval_dataset=d_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics2
)

print('Validating.')
res_eval = trainer.evaluate()
print(res_eval)

test_results = {}
print('Testing Haobo set, HPI-HC.')
y_test = trainer.predict(d_test_haoboset_hpihc)
test_results['haoboset_hpihc'] = y_test
print(y_test[2])

print('Testing ICD set, HPI-HC.')
y_test = trainer.predict(d_test_icdset_hpihc)
test_results['icdset_hpihc'] = y_test
print(y_test[2])

print('Testing Haobo labels, HPI-HC.')
y_test = trainer.predict(d_test_haobolabels_hpihc)
test_results['haobolabels_hpihc'] = y_test
print(y_test[2])

pl = pipeline("text-classification", model, tokenizer)
print(pl('Testing this out on a very scared old lady.'))
#fill_masker= FillMaskPipeline(model=model.to('cpu'), tokenizer=tokenizer)

#sentence1 = "Patient had significant GERD and was scheduled for a <mask> fundoplication procedure."
#sentence2 = "Patient was transferred from brigham and womens hospital and admitted to beth <mask> deaconess."

#print(fill_masker(sentence1))
#print(fill_masker(sentence2))
