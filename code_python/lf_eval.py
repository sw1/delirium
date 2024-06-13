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
params['n_train_epochs'] = 3 
params['lr'] = 5e-07 #7.53e-07
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
#d_train = d_train.train_test_split(test_size=0.95,shuffle=True,seed=s1)['train'] #subset data 

d_val = Dataset.from_dict(dat['val'])
#d_test_expert = Dataset.from_dict(dat['test_expert'])
d_test_icd = Dataset.from_dict(dat['test_icd'])
d_heldout = Dataset.from_dict(dat['heldout_expert'])
                           
# set up for designated pipeline                                      
if pl == 1: # finetune with repo model
    mod = os.path.join(out_finetune,'model')
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model1 = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    model2 = AutoModelForMaskedLM.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                              use_fast=True,max_length=seq_len)
    out_dir = out_finetune
elif pl == 2: # finetune with pretrained model
    mod = os.path.join(out_pretrain_finetune,'model')
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model1 = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    model2 = AutoModelForMaskedLM.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                              use_fast=True,max_length=seq_len)
    out_dir = out_pretrain_finetune
elif pl == 3: # finetune with pretrained model that used custom tokenizer
    mod = os.path.join(out_token_pretrain_finetune,'model')
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model1 = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    model2 = AutoModelForMaskedLM.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                              use_fast=True,max_length=seq_len)
    tokenizer_update = AutoTokenizer.from_pretrained(token_dir,
                                                     use_fast=True,max_length=seq_len)
    tokenizer.add_tokens(list(tokenizer_update.vocab))
    print('Length of updated tokenizer: %s' % len(tokenizer))
    dim1 = str(model1.get_input_embeddings())
    model1.resize_token_embeddings(len(tokenizer))
    model2.resize_token_embeddings(len(tokenizer))
    dim2 = str(model1.get_input_embeddings())
    print("Resizing model 1 embedding layer from %s to %s." % (dim1,dim2))
    dim2 = str(model2.get_input_embeddings())
    print("Resizing model 2 embedding layer from %s to %s." % (dim1,dim2))
    out_dir = out_token_pretrain_finetune
    
print('\nModel output directory:\n%s' % out_dir) 

tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

conf.hidden_dropout_prob=0.1
conf.classifier_dropout=0.1

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
#d_test_expert = d_test_expert.map(tokenize_dataset,batched=True,num_proc=cores)
d_test_icd = d_test_icd.map(tokenize_dataset,batched=True,num_proc=cores)
d_heldout = d_heldout.map(tokenize_dataset,batched=True,num_proc=cores)

# functions to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    scores = np.apply_along_axis(softmax, 1, logits)[:,1]
    preds = np.where(scores > 0.5,1,0)
    
    tn = sum([1 for i in range(len(preds)) if preds[i] == 0 and labels[i] == 0])
    fp = sum([1 for i in range(len(preds)) if preds[i] == 1 and labels[i] == 0])
    
    auc = evaluate.load('roc_auc').compute(references=labels, prediction_scores=scores)['roc_auc']
    acc = evaluate.load('accuracy').compute(predictions=preds, references=labels)['accuracy']
    prec = evaluate.load('precision').compute(predictions=preds, references=labels)['precision']
    rec = evaluate.load('recall').compute(predictions=preds, references=labels)['recall']
    f1 = evaluate.load('f1').compute(predictions=preds, references=labels)['f1']
    spec = tn/(tn + fp)
    b_acc = (rec + spec)/2

    return {'accuracy': acc, 'b_accuracy': b_acc,
            'f1': f1, 'auc': auc, 'precision': prec, 'recall': rec, 
            'batch_length': len(preds),'pred_positive': sum(preds), 'true_positive': sum(labels)}

print('Training groups.')
print([d_train['labels'].count(0),d_train['labels'].count(1)])

print('Validation groups.')
print([d_val['labels'].count(0),d_val['labels'].count(1)])

print('Testing icd groups.')
print([d_test_icd['labels'].count(0),d_test_icd['labels'].count(1)])

print('Heldout groups.')
print([d_heldout['labels'].count(0),d_heldout['labels'].count(1)])
class cTrainer(Trainer):
    pass
    
trainer = cTrainer(
    model=model1.to(device),
    eval_dataset=d_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

test_results = {}

print('Validating.')
res_eval = trainer.evaluate()
print(res_eval)
test_results['eval'] = res_eval

# performance adjusting for chunks since each chunk has the same label but
# is associated with a different section. hence will take max logistic
# score between chunks to yield final predicted label

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

pl1 = TextClassificationPipeline(model=model1.to('cpu'),tokenizer=tokenizer)
pl2 = FillMaskPipeline(model2.to('cpu'),tokenizer=tokenizer)

s1 = 'This frail old lady came in confused. Ended up needing to be oriented \
to place given her confusion. This happened after surgery.'
s2 = 'This lady has alzheimers. She otherwise is healthy and at her baseline \
is aoxtwo. After her surgery, she became aoxzero and was confused. She was \
discharged a few days later once she returned to baseline.'
s3 = 'This lady has alzheimers. She otherwise is healthy and at her baseline \
is aoxtwo. After her surgery, she was still aoxtwo. She had no issues during \
her stay. She slept well throughout the night. She remained afebrile and was \
eating well and ambulating. She was discharged that day.'

print(s1)
print(pl1(s1))

print(s2)
print(pl1(s2))

print(s3)
print(pl1(s3))

# tokenizer specific testing
print(pl2("Patient had significant GERD and was scheduled for a <mask> \
fundoplication procedure."))
print(pl2("Patient was transferred from brigham and womens hospital and \
admitted to beth <mask> deaconess."))

# pt type
print(pl2("A <mask> patient experienced delirium after a exploratory \
laparotomy."))
print(pl2("A <mask> patient experienced delirium after a laparoscopic \
appendectomy."))

# post op illness
print(pl2("A patient had a exploratory laparotomy and experienced <mask> \
a day after the procedure."))
print(pl2("A patient had a laparoscopic appendectomy and experienced <mask> \
a day after the procedure."))
print(pl2("A frail old patient coming from a nursing home had a exploratory \
laparotomy and experienced <mask> a day after the procedure."))
print(pl2("A frail old patient coming from a nursing home had a laparoscopic \
appendectomy and experienced <mask> a day after the procedure."))
print(pl2("A frail old patient coming from a nursing home had a hip \
replacement and experienced <mask> a day after the procedure."))

# seizure ms prompts
print(pl2("The patient suffered a <mask>."))
print(pl2("The patient experienced a <mask>."))
print(pl2("The patients episode was a <mask>."))
print(pl2("Patient with refractory <mask>."))
print(pl2("Patient will undergo elective <mask>."))

# slightly adapted prompts
print(pl2("The patients delirium episode was due to <mask>."))
print(pl2("The patients delirium episode was treated with <mask>."))
print(pl2("Patient has a high likelihood of delirium so <mask> was ordered."))


eval_fn = os.path.join(out_dir,'final_results.pkl')
print('\nSaving eval results to \n%s' % eval_fn)
with open(eval_fn, 'wb') as f:
    pickle.dump(test_results, f)
