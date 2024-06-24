import os
import sys
import gc
import argparse

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

from bitsandbytes.optim import Adam as adam

import wandb

import transformers
from transformers import (
    AutoTokenizer, pipeline, DataCollatorForLanguageModeling, 
    AutoModelForSequenceClassification, AdamW, AutoModelForMaskedLM, 
    AutoConfig, TrainingArguments, Trainer, TextClassificationPipeline,
    DataCollatorForLanguageModeling, FillMaskPipeline, LongformerModel,
    LongformerTokenizer, LongformerForMaskedLM,
    EarlyStoppingCallback, IntervalStrategy, PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
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
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(random.randint(1000, 9999))
os.environ["WANDB_PROJECT"]= 'lf_pretrain'
os.environ["WANDB_LOG_MODEL"] = 'false'

parser = argparse.ArgumentParser(description='Input for pretraining.')
parser.add_argument('-p','--pipeline',
                    choices=[1,2],
                    default=1,
                    type=int,
                    help='Type of pipeline: 1=pretrain (default), 2=pretrain tokenized')
parser.add_argument('-s','--seed',
                    #choices=range(1,99999),
                    default=str(random.randint(1,99999)),
                    type=int,
                    help='Seed for training')

args = parser.parse_args()

pl = args.pipeline
s = args.seed

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if device == 'cuda:0':
    gc.collect()
    torch.cuda.empty_cache()

work_dir = '/home/swolosz1/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer'
data_dir = os.path.join(work_dir,'data')
out_dir = os.path.join(work_dir,'out')
    
tbl_fn = 'tbl_chunked4096.csv.gz'
cores = 16 # cpu processess for processing data and tokenizing

params = dict()
params['seq_len'] = 4096 
params['s'] = s
params['n_grad_accum'] = 2
params['n_batch'] = 8
params['n_train_epochs'] = 10
params['lr'] = 3e-05 
params['warmup'] = 500
params['cycles'] = 2
params['w_decay'] = 0.01 
params['log_steps'] = 500 #50
params['save_multiplier'] = 5
    
token_dir = os.path.join(out_dir,'token')
pretrain_dir = os.path.join(out_dir,'pretrain')

out_pretrain = os.path.join(pretrain_dir,'model_pretrain')
out_token_pretrain = os.path.join(pretrain_dir,'model_token_pretrain')
out_token = os.path.join(token_dir,'custom_tokenizer.json')
    
dat = read_data(os.path.join(data_dir,tbl_fn),exp='pretrain',chunked=True)

d_train = Dataset.from_dict(dat['train'])
d_val = Dataset.from_dict(dat['val'])

# for testing
d_train = d_train.train_test_split(test_size=0.90,shuffle=True)['train'] #subset data for testing

mod = 'yikuan8/Clinical-Longformer' # repo clinical longformer
tokenizer = AutoTokenizer.from_pretrained(mod,fast=True)
conf = AutoConfig.from_pretrained(mod,gradient_checkpointing=False)
model = AutoModelForMaskedLM.from_pretrained(mod,config=conf)
#model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

# silence warning that seemingly isnt important
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

if pl == 1: # pretrain with repo model
    out_dir = out_pretrain
if pl == 2: # pretrain with custom tokenizer
    # obtain new tokens not in repo tokenizer then add them to repo tokenizer
    tokenizer_update = AutoTokenizer.from_pretrained(token_dir,fast=True)
    new_tokens = list(set(tokenizer_update.vocab.keys()) - set(tokenizer.vocab.keys()))
    tokenizer.add_tokens(new_tokens)
    print('Length of updated tokenizer: %s' % len(tokenizer))
    
    # resize model embeddings to accomidate new tokens
    dim1 = str(model.get_input_embeddings())
    model.resize_token_embeddings(len(tokenizer))
    dim2 = str(model.get_input_embeddings())
    print("Resizing model embedding layer from %s to %s." % (dim1,dim2))
    
    out_dir = out_token_pretrain


# create dir if doesnt exist
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

# dump params to have record
with open(os.path.join(out_dir,'params.dat'),'w') as f:
    for key, value in params.items():
        print(f"{key}: {value}", file=f)
    
# tokenizer function
def tokenize_function(data):

    data['text'] = [
        line for line in data['text'] if len(line) > 0 and not line.isspace()
    ]
    
    return tokenizer(
        data['text'],
        padding='max_length',
        truncation=True,
        max_length=params['seq_len'],
        return_special_tokens_mask=True,
    )

d_train = d_train.map(tokenize_function,
                      batched=True,
                      num_proc=cores,
                      remove_columns=['text'])
            
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

n_steps = math.floor(len(d_train)*params['n_train_epochs']/params['n_batch'])
n_steps_eval = math.floor(0.05*n_steps)

#optimizer = torch.optim.AdamW(model.parameters(),lr=params['lr'],weight_decay=params['w_decay'])
optimizer = adam(model.parameters(),lr=params['lr'],weight_decay=params['w_decay'])
scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                            num_warmup_steps=params['warmup'],
                                            num_training_steps=n_steps,
                                            num_cycles=params['cycles'],
                                           )

training_args = TrainingArguments(
    seed=s,
    
    disable_tqdm=False,
                                  
    do_train=True,
    do_eval=True,
    
    output_dir=out_dir,
    logging_dir=os.path.join(out_dir,'log'),
    overwrite_output_dir=True,
    logging_strategy='steps',
    logging_steps=params['log_steps'], #10, #n_steps_eval, 
    save_strategy='steps',
    save_steps=params['log_steps'] * params['save_multiplier'], #n_steps_eval, 
    save_total_limit=1,
    
    report_to='wandb',
    
    warmup_steps=params['warmup'],
    
    evaluation_strategy='steps',
    eval_steps=params['log_steps'], 
    
    #load_best_model_at_end=True,
    
    num_train_epochs=params['n_train_epochs'],
    
    learning_rate=params['lr'],
    weight_decay=params['w_decay'],
    
    fp16=False,
    dataloader_num_workers=cores, 
    
    gradient_checkpointing=True,
    gradient_accumulation_steps=params['n_grad_accum'], 
    
    per_device_train_batch_size=params['n_batch'],
    per_device_eval_batch_size=params['n_batch'],
)

# train with early stopping
trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=d_train,
    eval_dataset=d_val,
    tokenizer=tokenizer,
    #optimizers=(optimizer, scheduler),
    data_collator=data_collator,
    plugins=[DDPPlugin(find_unused_parameters=True)],
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
)

model.save_pretrained(os.path.join(out_dir,'model'))
trainer.save_model(os.path.join(out_dir,'model_trainer')) 

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
plt.savefig(os.path.join(out_dir,'figure1.png'))

fill_masker= FillMaskPipeline(model=model.to('cpu'),tokenizer=tokenizer)
print(fill_masker('Patient had significant GERD and was scheduled for a <mask> fundoplication procedure.'))
print(fill_masker('Patient was transferred from brigham and womens hospital and admitted to beth <mask> deaconess.'))
print(fill_masker('Patient was old and confused a day after surgery. She likely was experiencing <mask>.'))
print(fill_masker('A patient suffering from post operative delirium most likely also suffers from <mask>.'))
print(fill_masker('A patient suffering from post operative delirium most likely had the procedure <mask>.'))
