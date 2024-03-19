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

s1 = 1234 
seq_len = 4096
checkpoint = False
cp = 'checkpoint-50000'
n_epochs_cp = 15
n_epochs = 20 #5
no_punc = False

pipelines = ('pretrain repo model',
             'pretrain with custom tokenizer')

if len(sys.argv) > 1 and sys.argv[1] in ['1','2']:
    pipeline = int(sys.argv[1]) 
    print('Running pipeline %s: %s.' % (pipeline,pipelines[pipeline-1]))
else:
    sys.exit("\nSpecify a pipeline:\n\n \
        \t1: %s\n \
        \t2: %s\n" % pipelines)

work_dir = '/home/swolosz1/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer'
data_dir = os.path.join(work_dir,'data')
out_dir = os.path.join(work_dir,'out')
    
if no_punc:
    tbl_fn = 'tbl_to_python_231205.csv.gz'
    print('Fitting model without punctuation.')
else:
    tbl_fn = 'tbl_to_python_updated_chunked.csv.gz'
    out_dir = os.path.join(out_dir,'punc')
    print('Fitting model with punctuation.')
    
token_dir = os.path.join(out_dir,'token')
pretrain_dir = os.path.join(out_dir,'pretrain')
finetune_dir = os.path.join(out_dir,'finetune')
sweep_dir = os.path.join(out_dir,'sweep')
    
out_pretrain = os.path.join(pretrain_dir,'model_pretrain')
out_token_pretrain = os.path.join(pretrain_dir,'model_token_pretrain')
out_token = os.path.join(token_dir,'custom_tokenizer.json')
    
dat = read_data(os.path.join(data_dir,tbl_fn))

d_train = Dataset.from_dict(dat['train']).remove_columns(['id','hpi','labels','icd_sum','labels_h'])
#d_val = Dataset.from_dict(dat['val']).remove_columns(['id','hpi','labels','icd_sum','labels_h'])

mod = 'yikuan8/Clinical-Longformer'  
tokenizer = AutoTokenizer.from_pretrained(mod,fast=True)

if pipeline == 1: # pretrain with repo model
    out_dir = out_pretrain
if pipeline == 2: # pretrain with custom tokenizer
    #tokenizer = Tokenizer.from_file(out_token)
    #tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    #tokenizer.add_special_tokens({'pad_token':'[PAD]','unk_token':'[UNK]','mask_token':'[MASK]'})
    tokenizer_update = AutoTokenizer.from_pretrained(token_dir,fast=True)
    new_tokens = list(set(tokenizer_update.vocab.keys()) - set(tokenizer.vocab.keys()))
    tokenizer.add_tokens(new_tokens)
    print('Length of updated tokenizer: %s' % len(tokenizer))
    out_dir = out_token_pretrain

tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

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
    dim1 = str(model.get_input_embeddings())
    model.resize_token_embeddings(len(tokenizer))
    dim2 = str(model.get_input_embeddings())
    print("Resizing model embedding layer from %s to %s." % (dim1,dim2))

d_train = d_train.train_test_split(test_size=0.05,shuffle=True,seed=323)
d_train = DatasetDict({
    'train': d_train['train'],
    'val': d_train['test']}
)

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
            
print(d_train)

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

trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=d_train["train"],
    eval_dataset=d_train["val"],
    data_collator=data_collator,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

results = trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

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
#tokenizer.save_pretrained(os.path.join(out_dir,'log'))
plt.savefig(os.path.join(out_dir,'figure1.png'))


fill_masker= FillMaskPipeline(model=model.to('cpu'), tokenizer=tokenizer)

sentence1 = "Patient had significant GERD and was scheduled for a <mask> fundoplication procedure."
sentence2 = "Patient was transferred from brigham and womens hospital and admitted to beth <mask> deaconess."

print(fill_masker(sentence1))
print(fill_masker(sentence2))
