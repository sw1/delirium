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

def group_preds(predictions,labels):
    
    N = len(labels)
    res_dict = dict()
    
    for i in range(N):
        idnum = labels[i]
        label = predictions[1][i]
        pred = predictions[0][i].argmax()
        
        try:
            current = res_dict[idnum]['pred'].append(pred)
            
        except KeyError:
            res_dict[idnum] = dict()
            res_dict[idnum]['label'] = label
            res_dict[idnum]['pred'] = [pred]
            
    return(res_dict)

def majority_vote(res_dict):
    
    maj_dict = {'matches':dict(),
                'ties':[]}

    for k,v in res_dict.items():
        
        if len(res_dict[k]['pred']) == 1:
            
            if res_dict[k]['pred'] == res_dict[k]['label']:
                maj_dict['matches'][k] = [1,1] 
            else:
                maj_dict['matches'][k] = [0,1] 
        
        if len(res_dict[k]['pred']) > 1:
            
            n_str = len(res_dict[k]['pred'])
            maj = res_dict[k]['pred'].count(1)/len(res_dict[k]['pred'])
            
            if maj == 0.5:
                maj_dict['ties'].append(k)
            else:
                if maj > 0.5:
                    maj_dict['matches'][k] = [1,n_str]
                if maj < 0.5:
                    maj_dict['matches'][k] = [0,n_str]

    return(maj_dict)

def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(random.randint(1000, 9999))

s1 = 1234 
seq_len = 4096
balance_data = True
no_punc = False

pipelines = ('finetune repo model',
             'finetune pretrained model',
             'finetune pretrained model that used custom tokenizer')

exit_break = "\nSpecify a pipeline and table filename:\n\n \
        \t pipeline 1: %s\n \
        \t pipeline 2: %s\n \
        \t pipeline 3: %s\n" % pipelines

if len(sys.argv) == 3 and sys.argv[1] in ['1','2','3']:
    pipeline = int(sys.argv[1]) 
    tbl_fn = sys.argv[2]
    if 'chunk' in tbl_fn:
        chunked = True
    folder_fn = tbl_fn.replace('tbl_to_python_updated_','punc_').replace('.csv.gz','')
else:
    sys.exit(exit_break) 
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("\n\nNumber of devices: %s.\n \
    Device set to %s.\n\n" % (torch.cuda.device_count(),device))

work_dir = '/home/swolosz1/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer'
data_dir = os.path.join(work_dir,'data')
out_dir = os.path.join(work_dir,'out')

token_dir = os.path.join(out_dir,'token')
pretrain_dir = os.path.join(out_dir,'pretrain')
finetune_dir = os.path.join(out_dir,'finetune',folder_fn)
sweep_dir = os.path.join(out_dir,'sweep')

model_pretrain = os.path.join(pretrain_dir,'model_pretrain')
model_token_pretrain = os.path.join(pretrain_dir,'model_token_pretrain')

out_token = os.path.join(token_dir,'custom_tokenizer.json')
    
print('\nModel output filename:\n%s' % tbl_fn)    
    
dat = read_data(os.path.join(data_dir,tbl_fn))

out_finetune = os.path.join(finetune_dir,'final_model_finetune')
out_pretrain_finetune = os.path.join(finetune_dir,'final_model_pretrain_finetune')
out_token_pretrain_finetune = os.path.join(finetune_dir,'final_model_token_pretrain_finetune')

d_train = Dataset.from_dict(dat['train'])
#d_train = d_train.train_test_split(test_size=0.95,shuffle=True,seed=s1)['train'] #subset data
d_val = Dataset.from_dict(dat['val'])
d_test_haobo = Dataset.from_dict(dat['test_haobo'])
d_test_icd = Dataset.from_dict(dat['test_icd'])

if chunked:
    d_heldout = read_data(os.path.join(data_dir,'tbl_to_python_updated_chunked_treeheldout.csv.gz'))
else:
    d_heldout = read_data(os.path.join(data_dir,'tbl_to_python_updated_treeheldout.csv.gz'))
    
d_heldout = Dataset.from_dict(d_heldout['test_haobo'])

if pipeline == 1: # finetune with repo model
    mod = 'yikuan8/Clinical-Longformer' 
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained(mod,use_fast=True,max_length=seq_len)
    out_dir = out_finetune
elif pipeline == 2: # finetune with pretrained model
    mod = os.path.join(model_pretrain,'model')
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                              use_fast=True,max_length=seq_len)
    out_dir = out_pretrain_finetune
elif pipeline == 3: # finetune with pretrained model that used custom tokenizer
    mod = os.path.join(model_token_pretrain,'model')
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                              use_fast=True,max_length=seq_len)
    tokenizer_update = AutoTokenizer.from_pretrained(token_dir,
                                                     use_fast=True,max_length=seq_len)
    tokenizer.add_tokens(list(tokenizer_update.vocab))
    print('Length of updated tokenizer: %s' % len(tokenizer))
    dim1 = str(model.get_input_embeddings())
    model.resize_token_embeddings(len(tokenizer))
    dim2 = str(model.get_input_embeddings())
    print("Resizing model embedding layer from %s to %s." % (dim1,dim2))
    out_dir = out_token_pretrain_finetune

print('\nModel output directory:\n%s' % out_dir) 

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

print('Tokenizing training data.')
d_train = d_train.map(tokenize_dataset,batched=True,num_proc=16)
print('Tokenizing validation data.')
d_val = d_val.map(tokenize_dataset,batched=True,num_proc=16)
print('Tokenizing testing data.')
d_test_haoboset_hpihc = d_test_haobo.map(tokenize_dataset,batched=True,num_proc=16)
d_test_icdset_hpihc = d_test_icd.map(tokenize_dataset,batched=True,num_proc=16)
d_test_haoboset_hpi = d_test_haobo.remove_columns('text').rename_column('hpi','text').map(tokenize_dataset,batched=True,num_proc=16)
d_test_icdset_hpi = d_test_icd.remove_columns('text').rename_column('hpi','text').map(tokenize_dataset,batched=True,num_proc=16)
d_test_haobolabels_hpihc = d_test_haobo.remove_columns('labels').rename_column('labels_h','labels').map(tokenize_dataset,batched=True,num_proc=16)
d_test_haobolabels_hpi = d_test_haobo.remove_columns(['text','labels']).rename_column('hpi','text').rename_column('labels_h','labels').map(tokenize_dataset,batched=True,num_proc=16)
d_heldout_haobolabels_hpihc = d_heldout.remove_columns('labels').rename_column('labels_h','labels').map(tokenize_dataset,batched=True,num_proc=16)

if balance_data:
    print("Balancing training data.")
    
    positive_label = d_train.filter(lambda example: example['labels']==1, num_proc=16) 
    negative_label = d_train.filter(lambda example: example['labels']==0, num_proc=16)
    
    n_upsamp = len(negative_label) // len(positive_label)

    random.seed(10)
    seeds = random.sample(range(9999), n_upsamp)

    balanced_data = None
    for s in seeds:
        if balanced_data:
            balanced_data = concatenate_datasets([balanced_data, interleave_datasets([
                positive_label.shuffle(seed=s), 
                negative_label.shuffle(seed=s)
            ])])
        else:
            balanced_data = interleave_datasets([positive_label, negative_label])
    d_train = balanced_data
    
    class cTrainer(Trainer):
        pass
else:
    w = compute_class_weight(class_weight='balanced',
                             classes=np.unique(d_train['labels']),
                             y=d_train['labels']).tolist()
    
    print('Class weights.')
    print(w)
    
    class cTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute weighted loss
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
        
# s/p sweep
if pipeline == 1:
    n_train_epochs=3
    lr=8.4603e-07
    lab_smooth=0.2070
    w_decay=0.1133
if pipeline == 2:
    n_train_epochs=2
    lr=6.1845e-07
    lab_smooth=0.0004
    w_decay=0.0009
if pipeline == 3:
    n_train_epochs=3
    lr=7.9403e-07
    lab_smooth=0.0003
    w_decay=0.0127

training_args = TrainingArguments(seed=s1,
                                  
                                  disable_tqdm=False,
                                  
                                  do_train=True,
                                  do_eval=True,
                                  
                                  output_dir=out_dir,
                                  logging_dir=os.path.join(out_dir,'log'),
                                  overwrite_output_dir=True,
                                  logging_strategy='steps',
                                  logging_steps=250,
                                  save_strategy='steps',
                                  save_steps=5000,
                                  
                                  evaluation_strategy='steps',
                                  #max_steps=50,
                                  eval_steps=250,
                                  warmup_steps=500,
                                  
                                  load_best_model_at_end=True,
                                  metric_for_best_model='f1',
                                  greater_is_better=True,
                                  
                                  num_train_epochs=n_train_epochs, 
                                  learning_rate=lr, #7e-6,
                                  lr_scheduler_type='constant_with_warmup',
                                  
                                  label_smoothing_factor=lab_smooth,
                                  weight_decay=w_decay, 
                                  
                                  fp16=True,
                                  auto_find_batch_size=True,
                                  dataloader_num_workers=16, 
                                  gradient_accumulation_steps=8,
                                  eval_accumulation_steps=8,
                                  gradient_checkpointing=False,
                                  per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8,
                                  
                                 )

def compute_metrics1(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

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

print('Training groups.')
print([d_train['labels'].count(0),d_train['labels'].count(1)])

print('Validation groups.')
print([d_val['labels'].count(0),d_val['labels'].count(1)])

print('Testing ICD groups.')
print([d_test_icd['labels'].count(0),d_test_icd['labels'].count(1)])

print('Testing Haobo groups.')
print([d_test_haobo['labels'].count(0),d_test_haobo['labels'].count(1)])

print('Testing Heldout groups.')
print([d_heldout_haobolabels_hpihc['labels'].count(0),d_heldout_haobolabels_hpihc['labels'].count(1)])

trainer = cTrainer(
    model=model.to(device),
    args=training_args,
    train_dataset=d_train,
    eval_dataset=d_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics2,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
print('Training.')
result = trainer.train()
model.save_pretrained(os.path.join(out_dir,'model'))

res_eval = []
res_loss = []
e_s = None
t_s = None
for r in trainer.state.log_history:
    try:
        res_eval.append([r['step'],r['eval_loss'],r['eval_accuracy'],r['eval_f1']])
        e_loss = r['eval_loss']
        e_s = r['step']
        if e_s != None and t_s !=None:
            if e_s == t_s:
                res_loss.append([e_s,e_loss,t_loss])
    except KeyError:
        try:
            t_loss = r['loss']
            t_s = r['step']
            if e_s != None and t_s !=None:
                if e_s == t_s:
                    res_loss.append([e_s,e_loss,t_loss])
        except KeyError:
            next

df = pd.DataFrame(res_loss,columns=['step','eval_loss','train_loss'])
plt.figure()
plt.plot(df['step'],df['eval_loss'],label='eval_loss')
plt.plot(df['step'],df['train_loss'],label='train_loss')
plt.legend()
plt.savefig(os.path.join(out_dir,'figure1.png'))

df = pd.DataFrame(res_eval,columns=['step','loss','acc','f1'])
plt.figure()
plt.plot(df['step'],df['acc'],label='acc')
plt.plot(df['step'],df['f1'],label='f1')
plt.legend()
plt.savefig(os.path.join(out_dir,'figure2.png'))

test_results = {}
if chunked:
    
    print('Testing Haobo set, HPI-HC.')
    y_test = trainer.predict(d_test_haoboset_hpihc)
    grouped = group_preds(y_test,d_test_haobo['id'])
    out = majority_vote(grouped)
    acc = sum([r[0] for r in out['matches'].values()])/len(out['matches'])
    y_test = {'results':out,'accuracy':acc}
    print(acc)
    test_results['haoboset_hpihc'] = y_test

    print('Testing ICD set, HPI-HC.')
    y_test = trainer.predict(d_test_icdset_hpihc)
    grouped = group_preds(y_test,d_test_icd['id'])
    out = majority_vote(grouped)
    acc = sum([r[0] for r in out['matches'].values()])/len(out['matches'])
    y_test = {'results':out,'accuracy':acc}
    print(acc)
    test_results['icdset_hpihc'] = y_test

    print('Testing Haobo set, HPI only.')
    y_test = trainer.predict(d_test_haoboset_hpi)
    grouped = group_preds(y_test,d_test_haobo['id'])
    out = majority_vote(grouped)
    acc = sum([r[0] for r in out['matches'].values()])/len(out['matches'])
    y_test = {'results':out,'accuracy':acc}
    print(acc)
    test_results['haoboset_hpi'] = y_test
    
    print('Testing ICD set, HPI only.')
    y_test = trainer.predict(d_test_icdset_hpi)
    grouped = group_preds(y_test,d_test_icd['id'])
    out = majority_vote(grouped)
    acc = sum([r[0] for r in out['matches'].values()])/len(out['matches'])
    y_test = {'results':out,'accuracy':acc}
    print(acc)
    test_results['icdset_hpi'] = y_test

    print('Testing Haobo labels, HPI-HC.')
    y_test = trainer.predict(d_test_haobolabels_hpihc)
    grouped = group_preds(y_test,d_test_haobo['id'])
    out = majority_vote(grouped)
    acc = sum([r[0] for r in out['matches'].values()])/len(out['matches'])
    y_test = {'results':out,'accuracy':acc}
    print(acc)
    test_results['haobolabels_hpihc'] = y_test

    print('Testing Haobo labels, HPI only.')
    y_test = trainer.predict(d_test_haobolabels_hpi)
    grouped = group_preds(y_test,d_test_haobo['id'])
    out = majority_vote(grouped)
    acc = sum([r[0] for r in out['matches'].values()])/len(out['matches'])
    y_test = {'results':out,'accuracy':acc}
    print(acc)
    test_results['haobolabels_hpi'] = y_test

    print('Testing Heldout Haobo labels, HPI-HC.')
    y_test = trainer.predict(d_heldout_haobolabels_hpihc)
    grouped = group_preds(y_test,d_heldout['id'])
    out = majority_vote(grouped)
    acc = sum([r[0] for r in out['matches'].values()])/len(out['matches'])
    y_test = {'results':out,'accuracy':acc}
    print(acc)
    test_results['heldout'] = y_test
    
else:    
    
    print('Testing Haobo set, HPI-HC.')
    y_test = trainer.predict(d_test_haoboset_hpihc)
    test_results['haoboset_hpihc'] = y_test
    print(y_test[2])

    print('Testing ICD set, HPI-HC.')
    y_test = trainer.predict(d_test_icdset_hpihc)
    test_results['icdset_hpihc'] = y_test
    print(y_test[2])

    print('Testing Haobo set, HPI only.')
    y_test = trainer.predict(d_test_haoboset_hpi)
    test_results['haoboset_hpi'] = y_test
    print(y_test[2])

    print('Testing ICD set, HPI only.')
    y_test = trainer.predict(d_test_icdset_hpi)
    test_results['icdset_hpi'] = y_test
    print(y_test[2])

    print('Testing Haobo labels, HPI-HC.')
    y_test = trainer.predict(d_test_haobolabels_hpihc)
    test_results['haobolabels_hpihc'] = y_test
    print(y_test[2])

    print('Testing Haobo labels, HPI only.')
    y_test = trainer.predict(d_test_haobolabels_hpi)
    test_results['haobolabels_hpi'] = y_test
    print(y_test[2])

    print('Testing Heldout Haobo labels, HPI-HC.')
    y_test = trainer.predict(d_heldout_haobolabels_hpihc)
    test_results['heldout'] = y_test
    print(y_test[2])

with open(os.path.join(out_dir,'test_results.pkl'), 'wb') as f:
    pickle.dump(test_results, f)
