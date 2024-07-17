import os
import sys
import gc

import random
import numpy as np
from scipy.special import softmax
import pandas as pd
import math
import pickle 
import re

import matplotlib.pyplot as plt
from pynvml import *
from plotnine import *

import torch
from torch.nn import CrossEntropyLoss
import logging

import wandb

from transformers import (
    AutoTokenizer, DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    LongformerForSequenceClassification,
    AutoConfig, TrainingArguments,
    AutoModelForMaskedLM,
    Trainer, HfArgumentParser,
    EarlyStoppingCallback, set_seed,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
import tokenizers

import evaluate
from datasets import load_dataset, Dataset

from sklearn.utils import compute_class_weight
from sklearn.metrics import (precision_recall_fscore_support,
                             balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split

from typing import Optional
from dataclasses import dataclass, field

# import custom functions
from lf_functions import (
    read_data, softmax, compute_eval_metrics,
    balance_data, get_class_weights, check_and_save_params,
    pull_results, print_vars, process_log_history, final_preds,
)

@dataclass
class SweepArgs:
    """
    Arguements for sweeping through Longformer params.
    """
    
    train_method: str = field(default='finetune')
    threshold: Optional[int] = field(default=70)
    fraction: Optional[int] = field(default=100 )
    label: Optional[str] = field(default='only')
    pipeline: Optional[int] = field(default=1)
    sweep: Optional[bool] = field(default=False)
    folder_fn: Optional[str] = field(default=None)
    wandb_pn: Optional[str] = field(default=None)
    load_cp: Optional[str] = field(default=None)
    final_sweep: Optional[bool] = field(default=None)
    
    def __post_init__(self):
        valid_train_methods = ['pretrain','finetune']
        valid_thresholds = [60,70,80,90]
        valid_fractions = [20,35,50,75,100]
        valid_labels = ['icd','full','only','pseudo']
        valid_pipelines = [1,2,3]
        if self.train_method not in valid_train_methods:
            raise ValueError(f"Invalid value for train_method. Expected one of {valid_trains_methods}, but got {self_train_method}.")
        if self.threshold not in valid_thresholds:
            raise ValueError(f"Invalid value for threshold. Expected one of {valid_thresholds}, but got {self.threshold}.")
        if self.fraction not in valid_fractions:
            raise ValueError(f"Invalid value for fraction. Expected one of {valid_fractions}, but got {self.fraction}.")
        if self.label not in valid_labels:
            raise ValueError(f"Invalid value for label. Expected one of {valid_labels}, but got {self.label}.")
        if self.pipeline not in valid_pipelines:
            raise ValueError(f"Invalid value for pipeline. Expected one of {valid_pipelines}, but got {self.pipeline}.")
        assert not (self.sweep and self.folder_fn is None), 'If sweep is True, you must provide a folder_fn.'

@dataclass
class ModelArgs:
    """
    Arguements for Longformer finetuning.
    """
    folder_suffix: Optional[str] = field(default=None)
    input_table: Optional[str] = field(default='tbl.csv.gz')
    work_dir: Optional[str] = field(default='/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer')
    n_cores: Optional[int] = field(default=16)
    seed: Optional[int] = field(default=None)
    seq_len: Optional[int] = field(default=4096)
    num_labels: Optional[int] = field(default=2)
    n_grad_accum: Optional[int] = field(default=2)
    n_grad_accum_eval: Optional[int] = field(default=1)
    testing: Optional[bool] = field(default=False)
    n_batch: Optional[int] = field(default=32)
    n_batch_eval: Optional[int] = field(default=64)
    n_train_epochs: Optional[float] = field(default=4)
    lr: Optional[float] = field(default=8e-06 )
    warmup_ratio: Optional[float] = field(default=0.07)
    n_cycles: Optional[float] = field(default=0.0)
    label_smoothing: Optional[float] = field(default=0.0)
    do_hidden: Optional[float] = field(default=0.1)
    do_class: Optional[float] = field(default=0.1)
    w_decay: Optional[float] = field(default=0.01)
    f_log_steps: Optional[float] = field(default=0.01)
    save_multiplier: Optional[int] = field(default=2)
    f_subset_data: Optional[float] = field(default=None)
    log_steps: Optional[int] = field(default=None)

@dataclass
class TuneArgs:
    """
    Arguements for tuning Longformer params.
    """
    upsample: Optional[bool] = field(default=False)
    class_weights: Optional[bool] = field(default=True)
    class_weighting: Optional[float] = field(default=None)
    filter_keywords: Optional[bool] = field(default=True)
    group_by_len: Optional[bool] = field(default=True)
    pad_max_len: Optional[bool] = field(default=False)
    use_collator: Optional[bool] = field(default=True)
    out_dir: Optional[str] = field(default=None)
    overwrite_prompt: Optional[bool] = field(default=True)
    early_stopping: Optional[bool] = field(default=False)
    
    def __post_init__(self):
        
        print('\n')
        assert not (self.class_weights and self.upsample), 'Both class_weights and up_sample cannot be set to True.'
        assert not (self.group_by_len and self.pad_max_len), 'Both group_by_len and pad_max_len cannot be set to True.'
        
        if self.pad_max_len:
            if self.use_collator:
                self.pad_max_len = False
            else:
                self.pad_max_len = 'max_length'
        
        if self.group_by_len:
            if not self.use_collator:
                self.use_collator = True
                print(f"Enforcing use_collator to {self.use_collator} since group_by_len is {self.group_by_len}")


class wTrainer(Trainer):
    class_weights = None
    
    @classmethod
    def load_class_weights(cls, class_weights):
        cls.class_weights = class_weights
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(self.class_weights).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        
        outputs = model(**inputs)
        logits = outputs.get('logits')

        loss_fxn = CrossEntropyLoss(weight=self.class_weights) 
        loss = loss_fxn(logits.view(-1, self.model.config.num_labels),labels.view(-1))

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    scores = softmax(logits,axis=1)[:,1]
    preds = np.argmax(logits, axis=1)

    auc = evaluate.load('roc_auc').compute(references=labels, prediction_scores=scores)['roc_auc']
    acc = evaluate.load('accuracy').compute(predictions=preds, references=labels)['accuracy']
    prec = evaluate.load('precision').compute(predictions=preds, references=labels)['precision']
    rec = evaluate.load('recall').compute(predictions=preds, references=labels)['recall']
    f1 = evaluate.load('f1').compute(predictions=preds,references=labels)['f1']
    bacc = balanced_accuracy_score(y_true=labels,y_pred=preds)
    ppp = sum(preds)/len(preds)
    ptp = sum(labels)/len(labels)
    prop_diff = ppp-ptp
    score = math.sqrt(bacc / 0.5 / (abs(prop_diff) + 1))


    with open(os.path.join(main.out_dir,'eval_results.dat'), 'a+') as f:
        print(str(round(acc,2)) + '\t' + str(round(bacc,2)) + '\t' +
              str(round(f1,2)) + '\t' + str(round(auc,2)) + '\t' +
              str(round(prec,2)) + '\t' + str(round(rec,2)) + '\t' +
              str(len(preds)) + '\t' + str(sum(preds)) + '\t' +
              str(sum(labels)) + '\n',file=f)

    return {'run_name': main.folder_fn,
            'accuracy': acc, 'b_accuracy': bacc, 
            'f1': f1, 'auc': auc, 'precision': prec, 'recall': rec, 
            'prop_pred_positive': ppp, 'prop_true_positive': ptp,
            'prop_diff': prop_diff, 'score': score}

def init(model_args, sweep_args):
    os.environ['MASTER_PORT'] = str(random.randint(1000, 9999))
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\nCuda device: {device}\n")
    
    if device != 'cpu':
        gc.collect()
        torch.cuda.empty_cache()
    
    if model_args.testing:
        os.environ['WANDB_DISABLED'] = 'true'

    if model_args.seed is None:
        seed = random.randint(1,99999)
        print(f"\nNo seed provided. Seed set to {seed} for this run.")
        set_seed(seed)
    elif model_args.seed == 0:
        print(f"\nSeed set to {model_args.seed} hence no seed set for this run.")
        pass
    else:
        set_seed(model_args.seed)
    
    main.folder_fn = get_folder_fn(model_args,sweep_args)
    
    return device, main.folder_fn

def update_tokenizer(tokenizer,model,token_dir):
    tokenizer_update = AutoTokenizer.from_pretrained(token_dir,fast=True)
    new_tokens = list(set(tokenizer_update.vocab.keys()) - set(tokenizer.vocab.keys()))
    tokenizer.add_tokens(new_tokens)
    print('Length of updated tokenizer: %s' % len(tokenizer))
    # resize model embeddings to accomidate new tokens
    dim1 = str(model.get_input_embeddings())
    model.resize_token_embeddings(len(tokenizer))
    dim2 = str(model.get_input_embeddings())
    print("Resizing model embedding layer from %s to %s." % (dim1,dim2))

    return tokenizer, model

def get_components(sweep_args,model_args,out_dir,folder_fn):
        
        print(f"\nRunning {sweep_args.train_method} pipeline {sweep_args.pipeline}.")    
        
        if sweep_args.train_method == 'finetune':   
            
            train_dir = os.path.join(out_dir,'finetune',folder_fn) # needs a unique folder since labels/strf
            
            token_dir = os.path.join(out_dir,'token') # labels/rfst doenst apply, so one model
            model_pretrain = os.path.join(out_dir,'pretrain','model_pretrain') # loc for pretrained mod
            model_token_pretrain = os.path.join(out_dir,'pretrain','model_token_pretrain') # loc for custom tok and pretrained mod

            if sweep_args.pipeline == 1: # finetune with repo model
                if sweep_args.load_cp is None:
                    mod = 'yikuan8/Clinical-Longformer' # repo clinical lf
                    conf = AutoConfig.from_pretrained(mod,num_labels=model_args.num_labels)
                    conf.hidden_dropout_prob=model_args.do_hidden
                    conf.classifier_dropout=model_args.do_class
                    model = LongformerForSequenceClassification.from_pretrained(mod,config=conf)
                    tokenizer = AutoTokenizer.from_pretrained(mod,use_fast=True,max_length=model_args.seq_len)
                    out_dir = os.path.join(train_dir,'final_model_finetune') # location to output mod, pl1
                else:
                    conf = AutoConfig.from_pretrained(os.path.join(sweep_args.load_cp,'config.json'),num_labels=2)
                    model = LongformerForSequenceClassification.from_pretrained(sweep_args.load_cp,config=conf)
                    tokenizer = AutoTokenizer.from_pretrained(sweep_args.load_cp,use_fast=True,max_length=4096)
            elif sweep_args.pipeline == 2: # finetune with pretrained model
                mod = os.path.join(model_pretrain,'model_trainer') # pretrained mod
                conf = AutoConfig.from_pretrained(mod,num_labels=model_args.num_labels)
                conf.hidden_dropout_prob=model_args.do_hidden
                conf.classifier_dropout=model_args.do_class
                model = LongformerForSequenceClassification.from_pretrained(mod,config=conf)
                tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                                          use_fast=True,max_length=model_args.num_labels)
                out_dir = os.path.join(train_dir,'final_model_pretrain_finetune') # location to output mod, pl2
            elif sweep_args.pipeline == 3: # finetune with pretrained model that used custom tokenizer
                mod = os.path.join(model_token_pretrain,'model_trainer') # pretrained mod using custom tok
                conf = AutoConfig.from_pretrained(mod,num_labels=model_args.num_labels)
                conf.hidden_dropout_prob=model_args.do_hidden
                conf.classifier_dropout=model_args.do_class
                model = LongformerForSequenceClassification.from_pretrained(mod,config=conf)
                tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                                          use_fast=True,max_length=model_args.seq_len)
                tokenizer, model = update_tokenizer(tokenizer,model,token_dir)
                out_dir = os.path.join(train_dir,'final_model_token_pretrain_finetune') # location to output mod, pl3
        
        elif sweep_args.train_method == 'pretrain':
            
            train_dir = os.path.join(out_dir,'pretrain') # needs a unique folder since labels/strf
            token_dir = os.path.join(model_args.work_dir,'out','token') # labels/rfst doenst apply, so one model
            
            mod = 'yikuan8/Clinical-Longformer' # repo clinical longformer
            tokenizer = AutoTokenizer.from_pretrained(mod,use_fast=True,max_length=model_args.seq_len)
            conf = AutoConfig.from_pretrained(mod)
            model = AutoModelForMaskedLM.from_pretrained(mod,config=conf)
            
            if sweep_args.pipeline == 1: # pretrain with repo model
                out_dir = os.path.join(train_dir,'model_pretrain') # loc for pretrained mod
            if sweep_args.pipeline == 2: # pretrain with custom tokenizer
                tokenizer, model = update_tokenizer(tokenizer,model,token_dir)
                out_dir = os.path.join(train_dir,'model_token_pretrain') # loc for custom tok and pretrained mod

        return model, tokenizer, conf, out_dir

def get_data(model_args,sweep_args,tune_args):
    dat = read_data(os.path.join(model_args.work_dir,'data',model_args.input_table),
                    th=sweep_args.threshold,fr=sweep_args.fraction,exp=sweep_args.label)

    class_weights = None
    if sweep_args.train_method == 'finetune':
        if tune_args.class_weights:
            if tune_args.class_weighting == 1.0:
                print('\nCalculating class weights based on label counts.\n')
                class_weights = get_class_weights(dat['train'], model_args.num_labels)
            else:
                class_weights = torch.as_tensor([1-tune_args.class_weighting, tune_args.class_weighting])
            print(f"Class weights:\n0={class_weights[0]}\n1={class_weights[1]}") 

    if tune_args.filter_keywords:
        print('\nFiltering keywords from training set used in expert labeling.')
        for i in range(len(dat['train']['text'])):
                note = dat['train']['text'][i]
                note = note.replace('delirium','').replace('encephalopathy','')
                dat['train']['text'][i] = note
                
    dat['heldout_expert_filtered'] = dat['heldout_expert'] 
    for i in range(len(dat['heldout_expert_filtered']['text'])):
        note = dat['heldout_expert_filtered']['text'][i]
        note = note.replace('delirium','').replace('encephalopathy','')
        dat['heldout_expert_filtered']['text'][i] = note

    d_train = Dataset.from_dict(dat['train'])

    if model_args.testing:
        print(f"\nSubsetting training data to 5% of total.")
        d_train = d_train.train_test_split(test_size=0.95,shuffle=True)['train'] 

    d_val = Dataset.from_dict(dat['val'])

    if sweep_args.train_method == 'finetune':
        d_heldout_icd = Dataset.from_dict(dat['heldout_icd'])
        d_heldout_expert = Dataset.from_dict(dat['heldout_expert'])
        d_heldout_expert_filtered = Dataset.from_dict(dat['heldout_expert_filtered'])
    else:
        d_heldout_icd = d_heldout_expert = class_weights = None

    return d_train, d_val, d_heldout_icd, d_heldout_expert, d_heldout_expert_filtered, class_weights

def get_folder_fn(model_args,sweep_args):
    if sweep_args.train_method == 'finetune':   
        folder_fn = 'fit_' + sweep_args.label
        if sweep_args.pipeline == 'pseudo':
            folder_fn += '_th' + th + '_fr' + fr
        folder_fn += '_pl' + str(sweep_args.pipeline) 
        if model_args.folder_suffix is not None:
            folder_fn += '_' + model_args.folder_suffix 
    elif sweep_args.train_method == 'pretrain':
        folder_fn = 'fit_' + 'pl' + str(sweep_args.pipeline) 

    return(folder_fn)

def main(model_args, tune_args, sweep_args):
    print('\nRun args:')
    print_vars({**vars(model_args), **vars(tune_args), **vars(sweep_args)})
    
    device, main.folder_fn = init(model_args, sweep_args)
    main.out_dir = os.path.join(model_args.work_dir,'out')

    d_train, d_val, d_heldout_icd, d_heldout_expert, d_heldout_expert_filtered, class_weights = get_data(model_args,sweep_args,tune_args)
    model, tokenizer, conf, main.out_dir = get_components(sweep_args,model_args,main.out_dir,main.folder_fn)

    if tune_args.out_dir is not None:
        if sweep_args.sweep and sweep_args.folder_fn is not None:
            main.folder_fn = sweep_args.folder_fn
            main.out_dir = os.path.join(tune_args.out_dir,main.folder_fn)
        else:
            main.out_dir = os.path.join(tune_args.out_dir,'out')
            print(f"\nOverwriting default output directory to {main.out_dir}.")
    elif sweep_args.train_method == 'pretrain':
        try:
            os.makedirs(main.out_dir)
        except FileExistsError:
            print(f"\nOutput folder exists: {main.out_dir}.")
            sys.exit(1)
    else:   
        main.out_dir, main.folder_fn = check_and_save_params(model_args, tune_args, sweep_args, main.out_dir, main.folder_fn)

    def tokenize_dataset(data):
        data['text'] = [line for line in data['text'] if len(line) > 0 and not line.isspace()]

        return tokenizer(data['text'],padding=tune_args.pad_max_len,truncation=True,
                         max_length=model_args.seq_len,return_special_tokens_mask=True)

    if model_args.testing:
        n_steps = 5
        n_warmup = 0
        n_epochs = 1
        log_steps = 1
        save_strategy = 'no'
        save_steps = 0
    else:
        if sweep_args.train_method == 'finetune':
            n_steps = int(len(d_train) * model_args.n_train_epochs / model_args.n_batch / model_args.n_grad_accum)
            n_warmup = int(n_steps * model_args.warmup_ratio)
            n_epochs = model_args.n_train_epochs
            log_steps = int(n_steps * model_args.f_log_steps)
            log_steps = model_args.log_steps if model_args.log_steps is not None else log_steps
            save_strategy = 'steps'
            save_steps = log_steps * model_args.save_multiplier
        elif sweep_args.train_method == 'pretrain':
            n_steps = int(len(d_train) * model_args.n_train_epochs / model_args.n_batch / model_args.n_grad_accum)
            n_warmup = int(n_steps * model_args.warmup_ratio)
            n_epochs = model_args.n_train_epochs
            log_steps = int(n_steps // n_epochs)
            save_strategy = 'steps'
            save_steps = log_steps
    print(f"\nRun will be over {n_steps} training steps, {log_steps} evaluation steps, {n_warmup} warmup steps, and {n_epochs} epochs.")

    print('\nTokenizing training data.')
    d_train = d_train.map(tokenize_dataset,batched=True,num_proc=model_args.n_cores,remove_columns=['text'])
    print('Tokenizing validation data.')
    d_val = d_val.map(tokenize_dataset,batched=True,num_proc=model_args.n_cores,remove_columns=['text'])
    
    if sweep_args.train_method == 'finetune':
        print('Tokenizing testing data.')
        d_heldout_icd = d_heldout_icd.map(tokenize_dataset,batched=True,num_proc=model_args.n_cores,remove_columns=['text'])
        d_heldout_expert = d_heldout_expert.map(tokenize_dataset,batched=True,num_proc=model_args.n_cores,remove_columns=['text'])
        d_heldout_expert_filtered = d_heldout_expert.map(tokenize_dataset,batched=True,num_proc=model_args.n_cores,remove_columns=['text'])
        
        if tune_args.upsample:
            print('Upsampling training data.')
            d_train = balance_data(d_train,cores=model_args.n_cores)
            
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        if tune_args.group_by_len:
            data_collator.padding = 'longest'
        else:
            data_collator.padding = 'max_length'
            data_collator.max_length = model_args.seq_len
            
    elif sweep_args.train_method == 'pretrain':
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15,pad_to_multiple_of=8)
        
    optimizer = torch.optim.AdamW(model.parameters(),lr=model_args.lr,weight_decay=model_args.w_decay)
    if sweep_args.sweep:
        scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
                                                      num_warmup_steps=n_warmup)
    else:
        if model_args.n_cycles > 0.5:
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                           num_warmup_steps=n_warmup,
                                                                           num_training_steps=n_steps,
                                                                           num_cycles=model_args.n_cycles,
                                                                          )
        elif model_args.n_cycles == 0.5:
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=n_warmup,
                                                        num_training_steps=n_steps,
                                                        num_cycles=model_args.n_cycles,
                                                       )
        else:
            scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
                                                          num_warmup_steps=n_warmup)
            


    if sweep_args.load_cp is None:
        training_args = TrainingArguments(
            disable_tqdm = False,

            do_train = True,
            do_eval = True,

            output_dir = main.out_dir,
            logging_dir = os.path.join(main.out_dir,'log'),
            overwrite_output_dir = True,
            logging_strategy = 'steps',
            logging_steps = log_steps, 
            save_strategy = save_strategy,
            save_steps = save_steps,
            save_total_limit = 1,

            evaluation_strategy = 'steps',
            eval_steps = log_steps, 

            load_best_model_at_end = False,

            max_steps = n_steps if model_args.testing else -1,

            num_train_epochs = n_epochs,
            learning_rate = model_args.lr, 
            weight_decay = model_args.w_decay,  
            warmup_steps = n_warmup,

            dataloader_num_workers = model_args.n_cores, 
            dataloader_persistent_workers = True,
            dataloader_pin_memory = True,
            
            group_by_length = tune_args.group_by_len,

            tf32 = True,

            gradient_accumulation_steps = model_args.n_grad_accum,
            eval_accumulation_steps = model_args.n_grad_accum_eval,

            per_device_train_batch_size = model_args.n_batch, 
            per_device_eval_batch_size = model_args.n_batch_eval, 
        )
        
        if sweep_args.train_method == 'finetune':
            if sweep_args.sweep:
                training_args.load_best_model_at_end = False
                training_args.save_total_limit = 1
            else:
                training_args.load_best_model_at_end = True
            training_args.metric_for_best_model = 'b_accuracy'
            training_args.greater_is_better = True
            training_args.label_smoothing_factor = model_args.label_smoothing
            training_args.gradient_checkpointing = True
            training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
        elif sweep_args.train_method == 'pretrain':
            training_args.gradient_checkpointing = False
            #training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
            training_args.tf32 = True
            group_by_length = False
            #auto_find_batch_size_size=True
            #training_args.fp16 = True
            #training_args.optim = 'adafactor'
            #optimizer = None
            
        if sweep_args.sweep and sweep_args.final_sweep:
            training_args.metric_for_best_model = 'b_accuracy'
            training_args.greater_is_better = True
            training_args.load_best_model_at_end = True
            training_args.save_total_limit = 1
            
            
    else:
        training_args = torch.load(os.path.join(sweep_args.load_cp,'training_args.bin'))
        training_args.output_dir = main.out_dir
        training_args.logging_dir = os.path.join(main.out_dir,'log')
        training_args.num_train_epochs =  model_args.n_train_epochs
        training_args.warmup_steps = 0
            
    if not model_args.testing:
        training_args.report_to = 'wandb',
        training_args.run_name = main.folder_fn
        
    if sweep_args.train_method == 'finetune':
        print(f"\nTraining groups: 0={d_train['labels'].count(0)}, 1={d_train['labels'].count(1)}")
        print(f"Validation groups: 0={d_val['labels'].count(0)}, 1={d_val['labels'].count(1)}")
        print(f"Heldout icd groups: 0={d_heldout_icd['labels'].count(0)}, 1={d_heldout_icd['labels'].count(1)}")
        print(f"Heldout expert groups: 0={d_heldout_expert['labels'].count(0)}, 1={d_heldout_expert['labels'].count(1)}")
    
    print('\nTraining args:')
    print_vars(vars(training_args))
          
    if tune_args.class_weights and sweep_args.train_method == 'finetune':
        print('\nUsing weighted loss function.')
        class cTrainer(wTrainer):
            pass
        cTrainer.load_class_weights(class_weights)
    else:
        class cTrainer(Trainer):
            pass
      
    if sweep_args.load_cp is not None and sweep_args.sweep:
        print('\nLoading from checkpoint.')
        
        trainer = cTrainer(
            model = model,
            args = training_args,
            train_dataset = d_train,
            eval_dataset = d_val,
            compute_metrics = compute_metrics,
            data_collator = data_collator,
            tokenizer = tokenizer,
            optimizers = (optimizer,scheduler),
        )
        
        print('\nTrainer args:')
        print(trainer.args)
        
        trainer.train(resume_from_checkpoint=os.path.join(sweep_args.load_cp))
    else:
        trainer = cTrainer(
            model = model,
            args = training_args,
            train_dataset = d_train,
            eval_dataset = d_val,
            tokenizer = tokenizer,
            optimizers = (optimizer,scheduler),
        )

        if sweep_args.train_method == 'finetune':
            trainer.compute_metrics = compute_metrics
            if tune_args.use_collator:
                trainer.data_collator = data_collator
            # create headers for eval metric results file
            with open(os.path.join(main.out_dir,'eval_results.dat'), 'w') as f:
                print('acc\tb_acc\tf1\tauc\tprec\trec\tbatch_len\tpred_pod\ttrue_pos\n',file=f)
        elif sweep_args.train_method == 'pretrain':
            trainer.data_collator = data_collator            
         
        if tune_args.early_stopping:
            trainer.callbacks = [EarlyStoppingCallback(early_stopping_patience=5,early_stopping_threshold=0.005)]
            
        trainer.train()

    trainer.save_model(os.path.join(main.out_dir,'model_trainer')) 
    
    if sweep_args.train_method == 'finetune':
        if sweep_args.sweep:
            res_eval = trainer.evaluate()
            eval_b_acc = res_eval.get('eval_b_accuracy')
            eval_prop_diff = res_eval.get('eval_prop_diff')
            eval_score = res_eval.get('eval_score')
            eval_res = {'filter_keywords': tune_args.filter_keywords,
                        'th': sweep_args.threshold,
                        'lr': model_args.lr,
                        'w_decay': model_args.w_decay,
                        'eff_n_batch': model_args.n_batch * model_args.n_grad_accum,
                        'lab_smooth': model_args.label_smoothing,
                        'b_acc': eval_b_acc,
                        'prop_diff': eval_prop_diff,
                        'score': eval_score,
                        'cw': tune_args.class_weighting if tune_args.class_weights else 0.5,

            }
            eval_res = pd.DataFrame([eval_res], columns=eval_res.keys())
            eval_res.to_csv(os.path.join(main.out_dir,'eval_res.csv'),index=False)

            y_hat = trainer.predict(d_heldout_expert)
            y_hat_filtered = trainer.predict(d_heldout_expert_filtered)

            with open(os.path.join(main.out_dir,'heldout.pkl'), 'wb') as f:
                pickle.dump(y_hat, f)

            with open(os.path.join(main.out_dir,'heldout_filtered.pkl'), 'wb') as f:
                pickle.dump(y_hat_filtered, f)

            with open(os.path.join(main.out_dir,'mm_ids.txt'), 'w') as f:
                print("Heldout results.\n",file=f)
                for k, v in y_hat[2].items():
                    print(f"{k}: {v}",file=f)

                print("\nHeldout filtered results.\n",file=f)
                for k, v in y_hat_filtered[2].items():
                    print(f"{k}: {v}",file=f)

                print("\nHeldout results.\n")
                for k, v in y_hat[2].items():
                    print(f"{k}: {v}")

                print("\nHeldout filtered results.\n")
                for k, v in y_hat_filtered[2].items():
                    print(f"{k}: {v}")

            y_preds = np.argmax(y_hat[0], axis=1)
            mm_ids = [d_heldout_expert['id'][i] for i in range(len(y_preds)) if y_preds[i] != y_hat[1][i]]

            with open(os.path.join(main.out_dir,'mm_ids.txt'), 'w') as f:
                for id in mm_ids:
                    print(f'{id}\n',file=f)

            y_preds_filtered = np.argmax(y_hat_filtered[0], axis=1)
            mm_ids_filtered = [d_heldout_expert['id'][i] for i in range(len(y_preds_filtered)) if y_preds_filtered[i] != y_hat_filtered[1][i]]

            with open(os.path.join(main.out_dir,'mm_ids_filtered.txt'), 'w') as f:
                for id in mm_ids_filtered:
                    print(f'{id}\n',file=f)

        else:
            process_log_history(sweep_args,trainer.state.log_history,main.out_dir)

            final_preds(out_dir=main.out_dir,
                        args={**vars(model_args), **vars(tune_args), **vars(sweep_args)},
                        **{'heldout_icd': d_heldout_icd, 'heldout_expert': d_heldout_expert})


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArgs,TuneArgs,SweepArgs))
    model_args, tune_args, sweep_args = parser.parse_args_into_dataclasses()
    
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    
    if sweep_args.wandb_pn is not None:
        os.environ["WANDB_PROJECT"]= sweep_args.wandb_pn
    else:
        os.environ["WANDB_PROJECT"]= 'lf_' + sweep_args.train_method
                         
    os.environ["WANDB_LOG_MODEL"] = 'false'
    
    logging.getLogger('transformers').setLevel(logging.ERROR)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    main(model_args, tune_args, sweep_args)
