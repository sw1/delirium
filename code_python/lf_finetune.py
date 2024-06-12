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

# import custom functions
from lf_functions import *

os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(random.randint(1000, 9999))

pipelines = ('finetune repo model',
             'finetune pretrained model',
             'finetune pretrained model that used custom tokenizer')

exit_break = "\nSpecify pipeline, table filename, seeds (2), train_on_expert:\n\n \
        \t pipeline 1: %s\n \
        \t pipeline 2: %s\n \
        \t pipeline 3: %s\n\n \
        \t consider \n \
        \t\t dropout 0.1 \n \
        \t\t w_decay 0.0009-0.11\n \
        \t\t lab_smooth 0.0004-0.2 \n" % pipelines

# bash prompt to pass in params for fit and dir/filenames
# in cmd line pass in tbl name assuming tbl has 'chunked' in middle.
# this will append lf params to name and keep st params
# call should look like: 
# python lf_finetune.py tbl_to_python_expertupdate_chunked_rfst_majvote_th70_nfeat75.csv.gz 1 532 412 1
if len(sys.argv) == 6 and sys.argv[2] in ['1','2','3']:
    tbl_fn = sys.argv[1]
    pl = int(sys.argv[2]) 
    w_decay = 0.0002 
    lab_smooth = 0.00001 
    do_hidden = 0.1 # dropout in hidden layer
    do_class = 0.1 # dropout in classification layer
    s1 = int(sys.argv[3]) # seed dujring class balancing
    s2 = int(sys.argv[4]) # seed during training
    train_on_expert = int(sys.argv[5]) # whether to use remaining expert labels in training
    folder_fn = 'fit' + tbl_fn.replace('tbl_to_python_expertupdate','').replace('.csv.gz','')
    
    if 'rfst' in tbl_fn:
        st = True
    else:
        st = False
        
    if train_on_expert == 1:
        train_on_expert = True
else:
    sys.exit(exit_break) 
    
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('\n\nNumber of devices: %s.\n \
    Device set to %s.\n' % (torch.cuda.device_count(),device))

print('\nModel output folder name:\n%s\n' % folder_fn)    

# cpu processess for processing data and tokenizing
cores = 16
seq_len = 4096 
    
params = dict()
params['s1'] = s1
params['s2'] = s2
params['do_hidden'] = do_hidden
params['do_class'] = do_class
params['n_train_epochs'] = 3 #15 #3 
params['lr'] = 5e-07 #2e-06 #5e-06 #7.53e-07
params['w_decay'] = w_decay
params['lab_smooth'] = lab_smooth

# set directory locations
work_dir = '/home/swolosz1/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer'
data_dir = os.path.join(work_dir,'data')
out_dir = os.path.join(work_dir,'out')

token_dir = os.path.join(out_dir,'token') # labels/rfst doenst apply, so one model
pretrain_dir = os.path.join(out_dir,'pretrain') # labels/rfst doesnt apply, so one model
finetune_dir = os.path.join(out_dir,'finetune',folder_fn) # needs a unique folder since labels/strf

model_pretrain = os.path.join(pretrain_dir,'model_pretrain') # loc for pretrained mod
model_token_pretrain = os.path.join(pretrain_dir,'model_token_pretrain') # loc for custom tok and pretrained mod

out_token = os.path.join(token_dir,'custom_tokenizer.json') # location of custom tok
out_finetune = os.path.join(finetune_dir,'final_model_finetune') # location to output mod, pl1
out_pretrain_finetune = os.path.join(finetune_dir,'final_model_pretrain_finetune') # location to output mod, pl2
out_token_pretrain_finetune = os.path.join(finetune_dir,'final_model_token_pretrain_finetune') # location to output mod, pl3

# load data w/ function above
dat = read_data(os.path.join(data_dir,tbl_fn),train_on_expert=train_on_expert,st=st)

# split sets
d_train = Dataset.from_dict(dat['train'])

#d_train = d_train.train_test_split(test_size=0.95,shuffle=True,seed=s1)['train'] #subset data for testing

d_val = Dataset.from_dict(dat['val'])
#d_test_expert = Dataset.from_dict(dat['test_expert'])
d_test_icd = Dataset.from_dict(dat['test_icd'])
d_heldout = Dataset.from_dict(dat['heldout_expert'])

# set up for designated pipeline                                      
if pl == 1: # finetune with repo model
    mod = 'yikuan8/Clinical-Longformer' # repo clinical lf
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained(mod,use_fast=True,max_length=seq_len)
    out_dir = out_finetune
elif pl == 2: # finetune with pretrained model
    mod = os.path.join(model_pretrain,'model') # pretrained mod
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                              use_fast=True,max_length=seq_len)
    out_dir = out_pretrain_finetune
elif pl == 3: # finetune with pretrained model that used custom tokenizer
    mod = os.path.join(model_token_pretrain,'model') # pretrained mod using custom tok
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    # same approach to train tokenizer:
    # load repo tokenizer and newly trained custom tokenizer
    # then add new tokens from custom tok to repo tok
    # then update dimension size of mod
    tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                              use_fast=True,max_length=seq_len)
    tokenizer_update = AutoTokenizer.from_pretrained(token_dir,
                                                     use_fast=True,max_length=seq_len)
    new_tokens = list(set(tokenizer_update.vocab.keys()) - set(tokenizer.vocab.keys()))
    tokenizer.add_tokens(new_tokens)
    print('Length of updated tokenizer: %s' % len(tokenizer))
    dim1 = str(model.get_input_embeddings())
    model.resize_token_embeddings(len(tokenizer))
    dim2 = str(model.get_input_embeddings())
    print('Resizing model embedding layer from %s to %s.' % (dim1,dim2))
    out_dir = out_token_pretrain_finetune

print('\nModel output directory:\n%s\n' % out_dir) 

# create dir if doesnt exist
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

# dump params to have record
with open(os.path.join(out_dir,'params.dat'),'w') as f:
    for key, value in params.items():
        print(f"{key}: {value}", file=f)

# silence warning that seemingly isnt important
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

# tokenizer function
def tokenize_dataset(data):
    
    data['text'] = [
        line for line in data['text'] if len(line) > 0 and not line.isspace()
    ]
        
    return tokenizer(
        data['text'],
        padding='max_length', 
        truncation=True,
        max_length=seq_len,
        return_special_tokens_mask=True,
    )

print('Tokenizing training data.')
d_train = d_train.map(tokenize_dataset,batched=True,num_proc=cores)
print('Tokenizing validation data.')
d_val = d_val.map(tokenize_dataset,batched=True,num_proc=cores)
print('Tokenizing testing data.')
d_test_icd = d_test_icd.map(tokenize_dataset,batched=True,num_proc=cores)
d_heldout = d_heldout.map(tokenize_dataset,batched=True,num_proc=cores)

# balance minorty class via upsampling with replacement
print('Balancing training and validation data.')
d_train = balance_data(d_train,s=params['s1'],cores=cores)
d_val = balance_data(d_val,s=params['s1'],cores=cores)


conf.hidden_dropout_prob=params['do_hidden']
conf.classifier_dropout=params['do_class']
    
training_args = TrainingArguments(seed=params['s2'],
                                  
                                  disable_tqdm=False,
                                  
                                  do_train=True,
                                  do_eval=True,
                                  
                                  output_dir=out_dir,
                                  logging_dir=os.path.join(out_dir,'log'),
                                  overwrite_output_dir=True,
                                  logging_strategy='steps',
                                  logging_steps=500, #250,
                                  save_strategy='steps',
                                  save_steps=500, #5000,
                                  save_total_limit=1,
                                  
                                  evaluation_strategy='steps',
                                  #max_steps=50, # stopper for testing
                                  eval_steps=500, #250,
                                  warmup_steps=1000, #500,
                                  
                                  load_best_model_at_end=True,
                                  metric_for_best_model='f1',
                                  greater_is_better=True,
                                  
                                  num_train_epochs=params['n_train_epochs'], 
                                  learning_rate=params['lr'], #7e-6,
                                  lr_scheduler_type='constant_with_warmup',
                                  
                                  label_smoothing_factor=params['lab_smooth'],
                                  weight_decay=params['w_decay'], 
                                  
                                  fp16=True,
                                  auto_find_batch_size=True,
                                  dataloader_num_workers=cores, 
                                  gradient_accumulation_steps=8,
                                  eval_accumulation_steps=8,
                                  gradient_checkpointing=False,
                                  per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8,
                                  
                                 )

# functions to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    scores = np.apply_along_axis(softmax, 1, logits)[:,1]
    preds = np.where(scores > 0.5,1,0)
    
    auc = evaluate.load('roc_auc').compute(references=labels, prediction_scores=scores)['roc_auc']
    acc = evaluate.load('accuracy').compute(predictions=preds, references=labels)['accuracy']
    prec = evaluate.load('precision').compute(predictions=preds, references=labels)['precision']
    rec = evaluate.load('recall').compute(predictions=preds, references=labels)['recall']
    f1 = evaluate.load('f1').compute(predictions=preds, references=labels)['f1']

    return {'accuracy': acc, 'f1': f1, 'auc': auc, 'precision': prec, 'recall': rec, 
            'batch_length': len(preds),'pred_positive': sum(preds), 'true_positive': sum(labels)}

print('Training groups.')
print([d_train['labels'].count(0),d_train['labels'].count(1)])

print('Validation groups.')
print([d_val['labels'].count(0),d_val['labels'].count(1)])

print('Testing icd groups.')
print([d_test_icd['labels'].count(0),d_test_icd['labels'].count(1)])

print('Heldout groups.')
print([d_heldout['labels'].count(0),d_heldout['labels'].count(1)])

# trainer function with early stopping
class cTrainer(Trainer):
    pass

trainer = cTrainer(
    model=model.to(device),
    args=training_args,
    train_dataset=d_train,
    eval_dataset=d_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3,early_stopping_threshold=0.005)],
)

# make out dir if doesnt exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print('Training.')
result = trainer.train()
model.save_pretrained(os.path.join(out_dir,'model'))

# function to pull results
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

# loss figure
df = pd.DataFrame(res_loss,columns=['step','eval_loss','train_loss'])
plt.figure()
plt.plot(df['step'],df['eval_loss'],label='eval_loss')
plt.plot(df['step'],df['train_loss'],label='train_loss')
plt.legend()
plt.savefig(os.path.join(out_dir,'figure1.png'))

# performance figure
df = pd.DataFrame(res_eval,columns=['step','loss','acc','f1'])
plt.figure()
plt.plot(df['step'],df['acc'],label='acc')
plt.plot(df['step'],df['f1'],label='f1')
plt.legend()
plt.savefig(os.path.join(out_dir,'figure2.png'))

# performance adjusting for chunks since each chunk has the same label but
# is associated with a different section. hence will take max logistic
# score between chunks to yield final predicted label
test_results = {}

print('Testing icd labels.')
y_test = trainer.predict(d_test_icd)
print(compute_eval_metrics(y_test,d_test_icd['id'],method=1))
print(compute_eval_metrics(y_test,d_test_icd['id'],method=2))
test_results['icd'] = y_test

print('Heldout labels.')
y_test = trainer.predict(d_heldout)
print(compute_eval_metrics(y_test,d_heldout['id'],method=1))
print(compute_eval_metrics(y_test,d_heldout['id'],method=2))
test_results['heldout'] = y_test

# save results
with open(os.path.join(out_dir,'test_results.pkl'), 'wb') as f:
    pickle.dump(test_results, f)
