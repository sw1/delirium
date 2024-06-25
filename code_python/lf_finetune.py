import os
import sys
import shutil
import gc
import argparse

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
import torch.nn.functional as F

import wandb
#import accelerate

import transformers
from transformers import (
    AutoTokenizer, pipeline, DataCollatorForLanguageModeling, 
    AutoModelForSequenceClassification, AutoModelForMaskedLM, 
    AutoConfig, TrainingArguments, Trainer, TextClassificationPipeline,
    DataCollatorForLanguageModeling, FillMaskPipeline, LongformerModel,
    LongformerTokenizer, LongformerForMaskedLM,
    EarlyStoppingCallback, IntervalStrategy, get_cosine_schedule_with_warmup,
)
import tokenizers
from tokenizers import (
    decoders, models, normalizers, pre_tokenizers, processors,
    trainers, Tokenizer, AddedToken,
)
from transformers.integrations import *
import evaluate

import datasets
from datasets import (
    load_dataset, Dataset, load_metric, DatasetDict,
    concatenate_datasets, interleave_datasets,
)

from sklearn.utils import compute_class_weight
from sklearn.metrics import (precision_recall_fscore_support,
                             balanced_accuracy_score,
                             accuracy_score,
)
from sklearn.model_selection import train_test_split

# import custom functions
from lf_functions import *

os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(random.randint(1000, 9999))
os.environ["WANDB_PROJECT"]= 'lf_finetune'
os.environ["WANDB_LOG_MODEL"] = 'false'

parser = argparse.ArgumentParser(description='Input for finetuning.')
parser.add_argument('-t','--threshold',
                    choices=['90','80','70','60'],
                    type=str,
                    help='Threshold for pseudolabeling.)')
parser.add_argument('-f','--fraction',
                    choices=['100','75','50','35','20'],
                    default='100',
                    type=str,
                    help='Fraction of expert labeled data used for self-training. Only applicable to t=70.')
parser.add_argument('-l','--label',
                    required=True,
                    choices=['pseudo','full','only','icd'],
                    help='Labeling methodology: pseudo, full, only, icd')
parser.add_argument('-c','--chunked',
                    choices=['0','1'],
                    default='1',
                    type=bool,
                    help='Whether notes should be chunked: 1 (default), 0')
parser.add_argument('-p','--pipeline',
                    choices=[1,2,3],
                    default=1,
                    type=int,
                    help='Type of pipeline: 1=finetune (default), 2=finetune pretrained, 3=finetune pretrained tokenized')
parser.add_argument('-s','--seed',
                    #choices=range(1,99999),
                    default=str(random.randint(1,99999)),
                    type=int,
                    help='Seed for training')

args = parser.parse_args()

th = args.threshold
fr = args.fraction
lab = args.label
chunked = args.chunked
pl = args.pipeline
s = args.seed

if lab == 'pseudo':
    if th is None:
        sys.exit('If l=pseudo, then must specify a threshold.')
    if int(th) != 70 and int(fr) < 100:
        sys.exit('Only t=70 can take f values other than 100.')

folder_fn = 'fit_' + lab
if chunked:
    folder_fn += '_chunked'
if th is not None and fr is not None:
    folder_fn += '_th' + th + '_fr' + fr
folder_fn += '_pl' + str(pl) + '_s' + str(s) 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if device == 'cuda:0':
    gc.collect()
    torch.cuda.empty_cache()

print('\nModel output folder name:\n%s\n' % folder_fn)    

tbl_fn = 'tbl_chunked4096.csv.gz'
cores = 16 # cpu processess for processing data and tokenizing

params = dict()
params['seq_len'] = 4096 
params['s'] = s
params['n_grad_accum'] = 8
params['n_batch'] = 8
params['n_train_epochs'] = 3
params['lr'] = 2e-05 
params['warmup'] = 500
params['cycles'] = 0
params['w_decay'] = 0.01 
#params['lab_smooth'] = 0.1
params['do_hidden'] = 0.1 # dropout in hidden layer
params['do_class'] = 0.1 # dropout in classification layer
params['log_steps'] = 100 #50
params['save_multiplier'] = 2

# set directory locations
work_dir = '/home/swolosz1/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer'
data_dir = os.path.join(work_dir,'data')
out_dir = os.path.join(work_dir,'out')

# if testing
#out_dir = '/shared/anesthesia/wolosomething/scratch'

token_dir = os.path.join(out_dir,'token') # labels/rfst doenst apply, so one model
pretrain_dir = os.path.join(out_dir,'pretrain') # labels/rfst doesnt apply, so one model
finetune_dir = os.path.join(out_dir,'finetune',folder_fn) # needs a unique folder since labels/strf

model_pretrain = os.path.join(pretrain_dir,'model_pretrain') # loc for pretrained mod
model_token_pretrain = os.path.join(pretrain_dir,'model_token_pretrain') # loc for custom tok and pretrained mod

out_token = os.path.join(token_dir,'custom_tokenizer.json') # location of custom tok
out_finetune = os.path.join(finetune_dir,'final_model_finetune') # location to output mod, pl1
out_pretrain_finetune = os.path.join(finetune_dir,'final_model_pretrain_finetune') # location to output mod, pl2
out_token_pretrain_finetune = os.path.join(finetune_dir,'final_model_token_pretrain_finetune') # location to output mod, pl3

# load data 
dat = read_data(os.path.join(data_dir,tbl_fn),th=th,fr=fr,exp=lab,chunked=chunked)

# calculate class weights for balancing classes in loss
#class_weights = (1/pd.DataFrame(dat['train']).labels.value_counts(normalize=True).sort_index()).tolist()
class_weights = (pd.DataFrame(dat['train']).shape[0]/(pd.DataFrame(dat['train']).labels.value_counts(normalize=False) * 2).sort_index()).tolist()
class_weights = torch.tensor(class_weights)
class_weights = class_weights/class_weights.sum()
print('\n\nClass weight 0, 1:\n%s\n' % class_weights) 

# remove keywords used in expert labeling
for i in range(len(dat['train']['text'])):
    note = dat['train']['text'][i]
    note = note.replace('delirium','').replace('encephalopathy','')
    dat['train']['text'][i] = note
    
# split sets
d_train = Dataset.from_dict(dat['train'])
d_val = Dataset.from_dict(dat['val'])
d_heldout_icd = Dataset.from_dict(dat['heldout_icd'])
d_heldout_expert = Dataset.from_dict(dat['heldout_expert'])

# for testing
#d_train = d_train.train_test_split(test_size=0.95,shuffle=True,seed=s1)['train'] #subset data for testing

# set up for designated pipeline                                      
if pl == 1: # finetune with repo model
    mod = 'yikuan8/Clinical-Longformer' # repo clinical lf
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained(mod,use_fast=True,max_length=params['seq_len'])
    out_dir = out_finetune
elif pl == 2: # finetune with pretrained model
    mod = os.path.join(model_pretrain,'model') # pretrained mod
    conf = AutoConfig.from_pretrained(mod,num_labels=2,gradient_checkpointing=False)
    model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
    tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                              use_fast=True,max_length=params['seq_len'])
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
                                              use_fast=True,max_length=params['seq_len'])
    tokenizer_update = AutoTokenizer.from_pretrained(token_dir,
                                                     use_fast=True,max_length=params['seq_len'])
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
tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True

# tokenizer function
def tokenize_dataset(data):
    
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

n_steps = math.floor(len(d_train)*params['n_train_epochs']/params['n_batch']/params['n_grad_accum'])
n_steps_eval = math.floor(0.05*n_steps)

print('Tokenizing training data.')
d_train = d_train.map(tokenize_dataset,batched=True,num_proc=cores,remove_columns=['text'])
print('Tokenizing validation data.')
d_val = d_val.map(tokenize_dataset,batched=True,num_proc=cores,remove_columns=['text'])
print('Tokenizing testing data.')
d_heldout_icd = d_heldout_icd.map(tokenize_dataset,batched=True,num_proc=cores,remove_columns=['text'])
d_heldout_expert = d_heldout_expert.map(tokenize_dataset,batched=True,num_proc=cores,remove_columns=['text'])

# balance minorty class via upsampling with replacement
#if folder_fn == 'fit' or folder_fn == 'fit_chunked':
#print('Balancing training data.')
#d_train = balance_data(d_train,s=params['s1'],cores=cores)
#d_val = balance_data(d_val,s=params['s1'],cores=cores)

conf.hidden_dropout_prob=params['do_hidden']
conf.classifier_dropout=params['do_class']

optimizer = torch.optim.AdamW(model.parameters(),lr=params['lr'],weight_decay=params['w_decay'])
scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                            num_warmup_steps=params['warmup'],
                                            num_training_steps=n_steps,
                                            num_cycles=params['cycles'],
                                           )
    
training_args = TrainingArguments(
    seed=params['s'],
                                  
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
                                
    #max_steps=50, # stopper for testing
    
    evaluation_strategy='steps',
    eval_steps=params['log_steps'], #n_steps_eval, #250,
                                  
    load_best_model_at_end=True,
    metric_for_best_model='b_accuracy',
    greater_is_better=True,
    
    report_to='wandb',
    run_name=folder_fn, # + '_fewercycle_loadinglastmodel',
                                  
    num_train_epochs=params['n_train_epochs'],
    learning_rate=params['lr'], 
    #lr_scheduler_type='cosine_with_restarts',
    warmup_steps=params['warmup'],
                                  
    #label_smoothing_factor=params['lab_smooth'],
    weight_decay=params['w_decay'],                                 
                                  
    auto_find_batch_size=False, 
    dataloader_num_workers=cores, 
                                  
    fp16=False,
    bf16=False,
    
    gradient_checkpointing=True,
    gradient_accumulation_steps=params['n_grad_accum'],
    eval_accumulation_steps=params['n_grad_accum'],
                                  
                                  
    per_device_train_batch_size=params['n_batch'], 
    per_device_eval_batch_size=params['n_batch'], 
)

# create headers for eval metric results file
with open(os.path.join(out_dir,'eval_results.dat'), 'w') as f:
    print('acc\tb_acc\tf1\tauc\tprec\trec\tbatch_len\tpred_pod\ttrue_pos\n',file=f)

# functions to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    scores = np.apply_along_axis(softmax, 1, logits)[:,1]
    preds = np.argmax(logits, axis=1)
    
    auc = evaluate.load('roc_auc').compute(references=labels, prediction_scores=scores)['roc_auc']
    acc = evaluate.load('accuracy').compute(predictions=preds, references=labels)['accuracy']
    prec = evaluate.load('precision').compute(predictions=preds, references=labels)['precision']
    rec = evaluate.load('recall').compute(predictions=preds, references=labels)['recall']
    f1 = evaluate.load('f1').compute(predictions=preds, references=labels)['f1']
    bacc = balanced_accuracy_score(y_true=labels,y_pred=preds)
    
    with open(os.path.join(out_dir,'eval_results.dat'), 'a+') as f:
        print(str(round(acc,2)) + '\t' + str(round(bacc,2)) + '\t' +
              str(round(f1,2)) + '\t' + str(round(auc,2)) + '\t' +
              str(round(prec,2)) + '\t' + str(round(rec,2)) + '\t' +
              str(len(preds)) + '\t' + str(sum(preds)) + '\t' +
              str(sum(labels)) + '\n',file=f)

    return {'accuracy': acc, 'b_accuracy': bacc, 
            'f1': f1, 'auc': auc, 'precision': prec, 'recall': rec, 
            'batch_length': len(preds),'pred_positive': sum(preds), 'true_positive': sum(labels)}

print('Training groups.')
print([d_train['labels'].count(0),d_train['labels'].count(1)])

print('Validation groups.')
print([d_val['labels'].count(0),d_val['labels'].count(1)])

print('Heldout icd groups.')
print([d_heldout_icd['labels'].count(0),d_heldout_icd['labels'].count(1)])

print('Heldout expert groups.')
print([d_heldout_expert['labels'].count(0),d_heldout_expert['labels'].count(1)])

# trainer for class weighted loss
class wTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, 
                                              dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels').long()

        outputs = model(**inputs)

        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
    
# trainer function with early stopping
class cTrainer(Trainer):
    pass

trainer = wTrainer(
    class_weights=class_weights,
    model=model.to(device),
    args=training_args,
    train_dataset=d_train,
    eval_dataset=d_val,
    tokenizer=tokenizer,
    optimizers=(optimizer, scheduler),
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5,
                                       early_stopping_threshold=0.005)],
)

# make out dir if doesnt exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print('Training.')
trainer.train()

model.save_pretrained(os.path.join(out_dir,'model'))
trainer.save_model(os.path.join(out_dir,'model_trainer')) 

# function to pull results
res_eval = []
res_loss = []
res_lr = []
e_step = None
t_step = None
for r in trainer.state.log_history:
    try:
        res_eval.append([r['step'],r['eval_loss'],
                         r['eval_b_accuracy'],r['eval_f1']])
        e_loss = r['eval_loss']
        e_step = r['step']
        if e_step != None and t_step != None:
            if e_step == t_step:
                res_loss.append([e_step,e_loss,t_loss])
    except KeyError:
        try:
            t_loss = r['loss']
            t_step = r['step']
            res_lr.append([t_step,r['learning_rate']])
            if e_step != None and t_step != None:
                if e_step == t_step:
                    res_loss.append([e_step,e_loss])
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
df = pd.DataFrame(res_eval,columns=['step','loss','b_acc','f1'])
plt.figure()
plt.plot(df['step'],df['b_acc'],label='b_acc')
plt.plot(df['step'],df['f1'],label='f1')
plt.legend()
plt.savefig(os.path.join(out_dir,'figure2.png'))

# lr figure
df = pd.DataFrame(res_lr,columns=['step','lr'])
plt.figure()
plt.plot(df['step'],df['lr'])
plt.legend()
plt.savefig(os.path.join(out_dir,'figure3.png'))

# performance adjusting for chunks since each chunk has the same label but
# is associated with a different section. hence will take max logistic
# score between chunks to yield final predicted label
test_results = {}

print('Heldout icd labels.')
y_heldout_icd = trainer.predict(d_heldout_icd)
print(compute_eval_metrics(y_heldout_icd,d_heldout_icd['id'],method=1))
test_results['heldout_icd'] = y_heldout_icd
test_results['heldout_icd_ids'] = d_heldout_icd['id']

print('Heldout expert labels.')
y_heldout_expert = trainer.predict(d_heldout_expert)
print(compute_eval_metrics(y_heldout_expert,d_heldout_expert['id'],method=1))
test_results['heldout_expert'] = y_heldout_expert
test_results['heldout_expert_ids'] = d_heldout_expert['id']

# save testing results
with open(os.path.join(out_dir,'test_results.dat'), 'w') as f:
    print('Heldout ICD results:\n',file=f)
    print(compute_eval_metrics(y_heldout_icd,d_heldout_icd['id'],method=1), file=f) 
    print('\n\nHeldout expert results:\n',file=f)
    print(compute_eval_metrics(y_heldout_expert,d_heldout_expert['id'],method=1),file=f)

# save results
with open(os.path.join(out_dir,'test_results.pkl'), 'wb') as f:
    pickle.dump(test_results, f)
          
# save trainer_history
with open(os.path.join(out_dir,'log_history.pkl'), 'wb') as f:
    pickle.dump(trainer.state.log_history, f)
