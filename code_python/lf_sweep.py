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

import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import joblib

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
balance_data = True

pipelines = ('finetune repo model',
             'finetune pretrained model',
             'finetune pretrained model that used custom tokenizer')

if len(sys.argv) > 1 and sys.argv[1] in ['1','2','3']:
    pipeline = int(sys.argv[1]) 
    print('Running pipeline %s: %s.' % (pipeline,pipelines[pipeline-1]))
else:
    sys.exit("\nSpecify a pipeline:\n\n \
        \t1: %s\n \
        \t2: %s\n \
        \t3: %s\n" % pipelines)

    
work_dir = '/home/swolosz1/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer'
data_dir = os.path.join(work_dir,'data')
token_dir = os.path.join(work_dir,'out/token')
pretrain_dir = os.path.join(work_dir,'out/pretrain')
finetune_dir = os.path.join(work_dir,'out/finetune')
sweep_dir = os.path.join(work_dir,'out/sweep')

if balance_data:
    sweep_dir = os.path.join(sweep_dir,'balanced')
else:
    sweep_dir = os.path.join(sweep_dir,'weighted')

model_pretrain = os.path.join(pretrain_dir,'model_pretrain')
model_token_pretrain = os.path.join(pretrain_dir,'model_token_pretrain')

out_finetune = os.path.join(sweep_dir,'sweep_model_finetune')
out_pretrain_finetune = os.path.join(sweep_dir,'sweep_model_pretrain_finetune')
out_token_pretrain_finetune = os.path.join(sweep_dir,'sweep_model_token_pretrain_finetune')

out_token = os.path.join(token_dir,'custom_tokenizer.json')

dat = read_data(os.path.join(data_dir,'tbl_to_python_updated.csv.gz'))

d_train = Dataset.from_dict(dat['train'])
d_val = Dataset.from_dict(dat['val'])

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
    tokenizer = AutoTokenizer.from_pretrained(token_dir,use_fast=True,max_length=seq_len)
    out_dir = out_token_pretrain_finetune

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

if balance_data:
    print("Balancing training data.")
    
    positive_label = d_train.filter(lambda example: example['labels']==1, num_proc=8) 
    negative_label = d_train.filter(lambda example: example['labels']==0, num_proc=8)
    
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

def get_f1(trainer):
    for r in reversed(trainer.state.log_history):
        try:
            return r['eval_f1']
        except KeyError:
            next
            
def objective(trial):

    #conf_lf.hidden_dropout_prob=0.1 # 0.1
    #conf_lf.attention_probs_dropout_prob=0.1 # 0.1 # was off, changed 11/15
    #conf_lf.classifier_dropout=0.1 # 0.1
    
    training_args = TrainingArguments(

        disable_tqdm=False,
                                      
        do_train=True,
        do_eval=True,

        output_dir=out_dir,
        overwrite_output_dir=True,
        logging_dir=os.path.join(out_dir,'logs'),
        logging_strategy='steps',
        logging_steps=250,
        save_strategy='steps',
        save_steps=5000, #1000,

        evaluation_strategy="steps",
        #max_steps=50,
        eval_steps=250, #25,
        warmup_steps=500,

        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,

        num_train_epochs=trial.suggest_int("num_train_epochs", 2, 5),
        learning_rate=trial.suggest_float("learning_rate", 5e-7, 1e-6, log=True), 
        lr_scheduler_type='constant_with_warmup',
                                      
        label_smoothing_factor=trial.suggest_float("label_smoothing_factor",1e-5, 0.3, log=True),#0.1, #0.1,
        weight_decay=trial.suggest_float("weight_decay",1e-5, 0.3, log=True), #0.01, 

        fp16=True,
        auto_find_batch_size=True,
        dataloader_num_workers=16, 
        gradient_accumulation_steps=8,
        eval_accumulation_steps=8,
        gradient_checkpointing=False,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
                    
    )
    
    trainer = cTrainer(
        model=model.to(device),
        args=training_args,
        train_dataset=d_train,
        eval_dataset=d_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics2,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    trainer.train()
    
    return get_f1(trainer)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

storage = JournalStorage(JournalFileStorage(os.path.join(out_dir,"journal.log")))
sampler = optuna.samplers.RandomSampler()
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(
    direction="maximize", 
    study_name="predict_param_sweep",
    pruner=pruner,
    sampler=sampler,
    storage=storage,
    load_if_exists=True,
)
    
study.optimize(objective,n_trials=7,show_progress_bar=True)
        

print("Printing best value:")
print(study.best_value)
print("Printing best params:")
print(study.best_params)
print("Printing best trial:")
print(study.best_trial)

joblib.dump(study,os.path.join(out_dir,'study.pkl'))
