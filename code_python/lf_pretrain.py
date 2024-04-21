import os
import sys

import gzip
import csv
import random
import numpy as np
import pandas as pd
import math
import statistics

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
    EarlyStoppingCallback, IntervalStrategy, PreTrainedTokenizerFast,
)
import tokenizers
from tokenizers import (
    decoders, models, normalizers, pre_tokenizers, processors,
    trainers,Tokenizer,AddedToken,
)
from transformers.integrations import *
import evaluate

import datasets
from datasets import load_dataset, Dataset, load_metric, DatasetDict

from sklearn.utils import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

# import custom functions
from lf_functions import *

os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(random.randint(1000, 9999))

pipelines = ('pretrain repo model',
             'pretrain with custom tokenizer')

# bash command prompt to specify a pipeline:
# pretrain from repo or pretrain from custom tokenizer
if len(sys.argv) > 1 and sys.argv[1] in ['1','2']:
    pl = int(sys.argv[1]) 
    print('Running pipeline %s: %s.' % (pl,pipelines[pl-1]))
else:
    sys.exit("\nSpecify a pipeline:\n\n \
        \t1: %s\n \
        \t2: %s\n" % pipelines)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('\n\nNumber of devices: %s.\n \
    Device set to %s.\n' % (torch.cuda.device_count(),device))

s1 = 1234 
seq_len = 4096
checkpoint = False # whether to restart at checkpoint
cp = 'checkpoint-50000' # cp name if applicable
n_epochs_cp = 15 # n epochs after loading at checkpoint
n_epochs = 20 

work_dir = '/home/swolosz1/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer'
data_dir = os.path.join(work_dir,'data')
out_dir = os.path.join(work_dir,'out')
    
tbl_fn = 'tbl_to_python_expertupdate_chunked.csv.gz'
out_dir = os.path.join(out_dir,'punc')
    
token_dir = os.path.join(out_dir,'token')
pretrain_dir = os.path.join(out_dir,'pretrain')

out_pretrain = os.path.join(pretrain_dir,'model_pretrain')
out_token_pretrain = os.path.join(pretrain_dir,'model_token_pretrain')
out_token = os.path.join(token_dir,'custom_tokenizer.json')
    
dat = read_data(os.path.join(data_dir,tbl_fn),st=False,train_on_expert=False,finetuning=False)

# just using training data for tokenization
# no val samples which are saved strictly for validation during ft
# and no training expert labeled samples
d_train = Dataset.from_dict(dat['train'])
# creating small val set from training specificly for pretraining
d_train = d_train.train_test_split(test_size=0.05,shuffle=True,seed=323)
d_train = DatasetDict({
    'train': d_train['train'],
    'val': d_train['test']}
)


mod = 'yikuan8/Clinical-Longformer' # repo clinical longformer
tokenizer = AutoTokenizer.from_pretrained(mod,fast=True)

if pipeline == 1: # pretrain with repo model
    out_dir = out_pretrain
if pipeline == 2: # pretrain with custom tokenizer
    # obtain new tokens not in repo tokenizer then add them to repo tokenizer
    tokenizer_update = AutoTokenizer.from_pretrained(token_dir,fast=True)
    new_tokens = list(set(tokenizer_update.vocab.keys()) - set(tokenizer.vocab.keys()))
    tokenizer.add_tokens(new_tokens)
    print('Length of updated tokenizer: %s' % len(tokenizer))
    out_dir = out_token_pretrain
    
# silence warning that seemingly isnt important
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

# if loading from checkpoint
if checkpoint:
    cp_dir = os.path.join(out_dir,cp)
    conf = AutoConfig.from_pretrained(cp_dir)
    model = AutoModelForMaskedLM.from_pretrained(cp_dir,config=conf)
    n_epochs = n_epochs_cp
    print("Resuming from checkpoint %s.\nRunning over %d epochs.\n" % (cp_dir,n_epochs))
else:
    conf = AutoConfig.from_pretrained(mod,gradient_checkpointing=False)
    model = AutoModelForMaskedLM.from_pretrained(mod,config=conf)
    
if pipeline == 2:
    print("Length of trained tokenizer: %s" % len(tokenizer))
    # resize model embeddings to accomidate new tokens
    dim1 = str(model.get_input_embeddings())
    model.resize_token_embeddings(len(tokenizer))
    dim2 = str(model.get_input_embeddings())
    print("Resizing model embedding layer from %s to %s." % (dim1,dim2))


    # tokenizer function
def tokenize_function(data):

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

d_train = d_train.map(tokenize_function,
                      batched=True,
                      num_proc=16,
                      remove_columns=['text'])
            
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    seed=s1,
    
    disable_tqdm=False,
                                  
    do_train=True,
    do_eval=True,
    
    output_dir=out_dir,
    logging_dir=os.path.join(out_dir,'log'),
    overwrite_output_dir=True,
    logging_strategy='steps',
    logging_steps=10000,
    save_strategy='steps',
    save_steps=10000,
    
    warmup_steps=10000,
    evaluation_strategy='steps',
    eval_steps=10000,
    
    load_best_model_at_end=True,
    
    num_train_epochs=n_epochs,
    
    learning_rate=3e-5,
    weight_decay=0.01,
    
    optim='adamw_torch', #'adafactor',
    gradient_checkpointing=False,
    fp16=False,
    auto_find_batch_size=True,
    dataloader_num_workers=16, 
    #gradient_accumulation_steps=8,    
    per_device_train_batch_size=8,
    #per_device_eval_batch_size=8,
    
    resume_from_checkpoint = checkpoint,
)

# train with early stopping
trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=d_train['train'],
    eval_dataset=d_train['val'],
    data_collator=data_collator,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

results = trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# process fit object for results
res_eval = []
res_loss = []
e_s = None
t_s = None
for r in trainer.state.log_history:
    try:
        res_eval.append([r['step'],r['eval_loss']])
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

model.save_pretrained(os.path.join(out_dir,'model'))
plt.savefig(os.path.join(out_dir,'figure1.png'))

fill_masker= FillMaskPipeline(model=model.to('cpu'),tokenizer=tokenizer)
print(fill_masker('Patient had significant GERD and was scheduled for a <mask> fundoplication procedure.'))
print(fill_masker('Patient was transferred from brigham and womens hospital and admitted to beth <mask> deaconess.'))
print(fill_masker('Patient was old and confused a day after surgery. She likely was experiencing <mask>.'))
print(fill_masker('A patient suffering from post operative delirium most likely also suffers from <mask>.'))
print(fill_masker('A patient suffering from post operative delirium most likely had the procedure <mask>.'))
