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
import torch.nn.functional as F

import wandb

from trl import SFTTrainer

from transformers import (
    AutoTokenizer, DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    AutoModelForSequenceClassification, 
    AutoConfig, TrainingArguments,
    AutoModelForMaskedLM,
    Trainer, HfArgumentParser,
    EarlyStoppingCallback, set_seed,
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
    n_steps_testing: Optional[int] = field(default=None)
    n_batch: Optional[int] = field(default=32)
    n_batch_eval: Optional[int] = field(default=64)
    n_train_epochs: Optional[int] = field(default=4)
    lr: Optional[float] = field(default=8e-06 )
    warmup_ratio: Optional[float] = field(default=0.07)
    n_cycles: Optional[float] = field(default=0.5)
    label_smoothing: Optional[float] = field(default=0.0)
    do_hidden: Optional[float] = field(default=0.1)
    do_class: Optional[float] = field(default=0.1)
    w_decay: Optional[float] = field(default=0.01)
    f_log_steps: Optional[float] = field(default=0.01)
    save_multiplier: Optional[int] = field(default=2)
    f_subset_data: Optional[float] = field(default=None)
    
@dataclass
class TuneArgs:
    """
    Arguements for tuning Longformer params.
    """
    upsample: Optional[bool] = field(default=False)
    class_weights: Optional[bool] = field(default=True)
    filter_keywords: Optional[bool] = field(default=True)
    group_by_len: Optional[bool] = field(default=True)
    pad_max_len: Optional[bool] = field(default=False)
    use_collator: Optional[bool] = field(default=True)
    out_dir: Optional[str] = field(default=None)
    override_prompt: Optional[bool] = field(default=True)
    
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
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    scores = softmax(logits,axis=1)[:,1]
    preds = np.argmax(logits, axis=1)

    auc = evaluate.load('roc_auc').compute(references=labels, prediction_scores=scores)['roc_auc']
    acc = evaluate.load('accuracy').compute(predictions=preds, references=labels)['accuracy']
    prec = evaluate.load('precision').compute(predictions=preds, references=labels)['precision']
    rec = evaluate.load('recall').compute(predictions=preds, references=labels)['recall']
    f1 = evaluate.load('f1').compute(predictions=preds, references=labels)['f1']
    bacc = balanced_accuracy_score(y_true=labels,y_pred=preds)

    with open(os.path.join(main.out_dir,'eval_results.dat'), 'a+') as f:
        print(str(round(acc,2)) + '\t' + str(round(bacc,2)) + '\t' +
              str(round(f1,2)) + '\t' + str(round(auc,2)) + '\t' +
              str(round(prec,2)) + '\t' + str(round(rec,2)) + '\t' +
              str(len(preds)) + '\t' + str(sum(preds)) + '\t' +
              str(sum(labels)) + '\n',file=f)

    return {'run_name': main.folder_fn,
            'class_weights': tune_args.class_weights,
            'upsample': tune_args.upsample,'accuracy': acc, 'b_accuracy': bacc, 
            'f1': f1, 'auc': auc, 'precision': prec, 'recall': rec, 
            'batch_length': len(preds),'pred_positive': sum(preds), 'true_positive': sum(labels)}

def init(model_args, sweep_args):
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(random.randint(1000, 9999))
    os.environ["WANDB_PROJECT"]= 'lf_finetune'
    os.environ["WANDB_LOG_MODEL"] = 'false'
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\nCuda device: {device}")
    
    if device != 'cpu':
        gc.collect()
        torch.cuda.empty_cache()
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if model_args.n_steps_testing is not None:
        os.environ['WANDB_DISABLED'] = 'true'

    if model_args.seed is None:
        set_seed(random.randint(1,99999))
    else:
        set_seed(model_args.seed)
    
    if sweep_args.train_method == 'finetune':   
        main.folder_fn = 'fit_' + sweep_args.label
        if sweep_args.pipeline == 'pseudo':
            main.folder_fn += '_th' + th + '_fr' + fr
        main.folder_fn += '_pl' + str(sweep_args.pipeline) 
        if model_args.folder_suffix is not None:
            main.folder_fn += '_' + model_args.folder_suffix 
        print('\nModel output folder name:\n%s\n' % main.folder_fn)   
    else:
        main.folder_fn = None
    
    return device, main.folder_fn

def get_components(sweep_args,model_args,out_dir,folder_fn):
        
        print(f"\nRunning {sweep_args.train_method} pipeline {sweep_args.pipeline}.")    
        
        if sweep_args.train_method == 'finetune':   
            
            train_dir = os.path.join(out_dir,'finetune',folder_fn) # needs a unique folder since labels/strf
            
            if sweep_args.pipeline == 1: # finetune with repo model
                mod = 'yikuan8/Clinical-Longformer' # repo clinical lf
                conf = AutoConfig.from_pretrained(mod,num_labels=model_args.num_labels)
                model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
                tokenizer = AutoTokenizer.from_pretrained(mod,use_fast=True,max_length=model_args.seq_len)
                out_dir = os.path.join(train_dir,'final_model_finetune') # location to output mod, pl1
            elif sweep_args.pipeline == 2: # finetune with pretrained model
                mod = os.path.join(model_pretrain,'model') # pretrained mod
                conf = AutoConfig.from_pretrained(mod,num_labels=model_args.num_labels)
                model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
                tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                                          use_fast=True,max_length=model_args.num_labels)
                out_dir = os.path.join(train_dir,'final_model_pretrain_finetune') # location to output mod, pl2
            elif sweep_args.pipeline == 3: # finetune with pretrained model that used custom tokenizer
                mod = os.path.join(model_token_pretrain,'model') # pretrained mod using custom tok
                conf = AutoConfig.from_pretrained(mod,num_labels=model_args.num_labels)
                model = AutoModelForSequenceClassification.from_pretrained(mod,config=conf)
                # same approach to train tokenizer:
                # load repo tokenizer and newly trained custom tokenizer
                # then add new tokens from custom tok to repo tok
                # then update dimension size of mod
                tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                                          use_fast=True,max_length=model_args.seq_len)
                tokenizer_update = AutoTokenizer.from_pretrained(token_dir,
                                                                 use_fast=True,max_length=model_args.seq_len)
                new_tokens = list(set(tokenizer_update.vocab.keys()) - set(tokenizer.vocab.keys()))
                tokenizer.add_tokens(new_tokens)
                print('Length of updated tokenizer: %s' % len(tokenizer))
                dim1 = str(model.get_input_embeddings())
                model.resize_token_embeddings(len(tokenizer))
                dim2 = str(model.get_input_embeddings())
                print('Resizing model embedding layer from %s to %s.' % (dim1,dim2))
                out_dir = os.path.join(train_dir,'final_model_token_pretrain_finetune') # location to output mod, pl3
        
        elif sweep_args.train_method == 'pretrain':
            
            train_dir = os.path.join(out_dir,'pretrain') # needs a unique folder since labels/strf
            token_dir = os.path.join(model_args.work_dir,'out','token') # labels/rfst doenst apply, so one model
            
            if pl == 1: # pretrain with repo model
                out_dir = os.path.join(train_dir,'model_pretrain') # loc for pretrained mod
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
                out_dir = os.path.join(train_dir,'model_token_pretrain') # loc for custom tok and pretrained mod

            
        return model, tokenizer, conf, out_dir
    
def get_data(model_args,sweep_args,tune_args):
    dat = read_data(os.path.join(model_args.work_dir,'data',model_args.input_table),
                    th=sweep_args.threshold,fr=sweep_args.fraction,exp=sweep_args.label)

    class_weights = None
    if sweep_args.train_method == 'finetune':
        if tune_args.class_weights:
            class_weights = get_class_weights(dat['train'], model_args.num_labels)
            print(f"Class weights:\n0={class_weights[0]}\n1={class_weights[1]}") 

    if tune_args.filter_keywords:
        print('\nFiltering keywords used in expert labeling.')
        for i in range(len(dat['train']['text'])):
            note = dat['train']['text'][i]
            note = note.replace('delirium','').replace('encephalopathy','')
            dat['train']['text'][i] = note

    d_train = Dataset.from_dict(dat['train'])

    if model_args.f_subset_data is not None:
        print(f"Subsetting training data to {round(model_args.f_subset_data * 100)}% of total.")
        d_train = d_train.train_test_split(test_size=1-model_args.f_subset_data,shuffle=True)['train'] 

    d_val = Dataset.from_dict(dat['val'])

    if sweep_args.train_method == 'finetune':
        d_heldout_icd = Dataset.from_dict(dat['heldout_icd'])
        d_heldout_expert = Dataset.from_dict(dat['heldout_expert'])
    else:
        d_heldout_icd = d_heldout_expert = class_weights = None

    return d_train, d_val, d_heldout_icd, d_heldout_expert, class_weights

def main(model_args, tune_args, sweep_args):
    print('\nRun args:')
    print_vars({**vars(model_args), **vars(tune_args), **vars(sweep_args)})
    
    device, main.folder_fn = init(model_args, sweep_args)
    main.out_dir = os.path.join(model_args.work_dir,'out')

    d_train, d_val, d_heldout_icd, d_heldout_expert, class_weights = get_data(model_args,sweep_args,tune_args)
    model, tokenizer, conf, main.out_dir = get_components(sweep_args,model_args,main.out_dir,main.folder_fn)
  
    if tune_args.out_dir is not None:
        main.out_dir = os.path.join(tune_args.out_dir,'out')
        print(f"Overriding default output directory to {main.out_dir}.")
    else:   
        main.out_dir, main.folder_fn = check_and_save_params(model_args, tune_args, sweep_args, main.out_dir, main.folder_fn)

    def tokenize_dataset(data):
        data['text'] = [line for line in data['text'] if len(line) > 0 and not line.isspace()]

        return tokenizer(data['text'],padding=tune_args.pad_max_len,truncation=True,
                         max_length=model_args.seq_len,return_special_tokens_mask=True)

    if model_args.n_steps_testing is None:
        n_steps = math.floor(len(d_train)*model_args.n_train_epochs/model_args.n_batch/model_args.n_grad_accum)
        n_warmup = int(round(n_steps/model_args.warmup_ratio))
        n_epochs = model_args.n_train_epochs
        log_steps = int(round(n_steps * model_args.f_log_steps))
        log_steps = 100 if log_steps > 100 else log_steps
    else:
        n_steps = model_args.n_steps_testing
        n_warmup = 0
        n_epochs = 1
        log_steps = round(n_steps)/2
        print(f"Performing a testing run with {n_steps} training steps, {log_steps} evaluation steps, {n_warmup} steps, and {n_epochs} epochs.")

    print('\nTokenizing training data.')
    d_train = d_train.map(tokenize_dataset,batched=True,num_proc=model_args.n_cores,remove_columns=['text'])
    print('Tokenizing validation data.')
    d_val = d_val.map(tokenize_dataset,batched=True,num_proc=model_args.n_cores,remove_columns=['text'])
    
    if sweep_args.train_method == 'finetune':
        print('Tokenizing testing data.')
        d_heldout_icd = d_heldout_icd.map(tokenize_dataset,batched=True,num_proc=model_args.n_cores,remove_columns=['text'])
        d_heldout_expert = d_heldout_expert.map(tokenize_dataset,batched=True,num_proc=model_args.n_cores,remove_columns=['text'])

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
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        
    optimizer = torch.optim.AdamW(model.parameters(),lr=model_args.lr,weight_decay=model_args.w_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=n_warmup,
                                                num_training_steps=n_steps,
                                                num_cycles=model_args.n_cycles,
                                               )
    
    conf.hidden_dropout_prob=model_args.do_hidden
    conf.classifier_dropout=model_args.do_class

    training_args = TrainingArguments(
        disable_tqdm = False,

        do_train = True,
        do_eval = True,

        output_dir = main.out_dir,
        logging_dir = os.path.join(main.out_dir,'log'),
        overwrite_output_dir = True,
        logging_strategy = 'steps',
        logging_steps = log_steps, 
        save_strategy = 'steps',
        save_steps = log_steps * model_args.save_multiplier,
        save_total_limit = 1,

        evaluation_strategy = 'steps',
        eval_steps = log_steps, 
        
        load_best_model_at_end = False,
        
        num_train_epochs = n_epochs,
        learning_rate = model_args.lr, 
        weight_decay = model_args.w_decay,  
        warmup_steps = n_warmup,

        auto_find_batch_size = False, 
        dataloader_num_workers = model_args.n_cores, 
        dataloader_persistent_workers = True,
        dataloader_pin_memory = True,

        tf32 = True,

        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {'use_reentrant':False},
        gradient_accumulation_steps = model_args.n_grad_accum,
        eval_accumulation_steps = model_args.n_grad_accum_eval,

        per_device_train_batch_size = model_args.n_batch, 
        per_device_eval_batch_size = model_args.n_batch_eval, 
    )
    
    if model_args.n_steps_testing is None:
        training_args.report_to = 'wandb',
        training_args.run_name = main.folder_fn

    if sweep_args.train_method == 'finetune':
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = 'b_accuracy'
        training_args.greater_is_better = True
        training_args.label_smoothing_factor = model_args.label_smoothing
        
        print(f"\nTraining groups: 0={d_train['labels'].count(0)}, 1={d_train['labels'].count(1)}")
        print(f"Validation groups: 0={d_val['labels'].count(0)}, 1={d_val['labels'].count(1)}")
        print(f"Heldout icd groups: 0={d_heldout_icd['labels'].count(0)}, 1={d_heldout_icd['labels'].count(1)}")
        print(f"Heldout expert groups: 0={d_heldout_expert['labels'].count(0)}, 1={d_heldout_expert['labels'].count(1)}")
        
    training_args.group_by_length = tune_args.group_by_len
    
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
        trainer.callbacks = [EarlyStoppingCallback(early_stopping_patience=15,early_stopping_threshold=0.005)]
        if tune_args.use_collator:
            trainer.data_collator = data_collator

        # create headers for eval metric results file
        with open(os.path.join(main.out_dir,'eval_results.dat'), 'w') as f:
            print('acc\tb_acc\tf1\tauc\tprec\trec\tbatch_len\tpred_pod\ttrue_pos\n',file=f)
    elif sweep_args.train_method == 'pretrain':
        trainer.data_collator = data_collator
            
    trainer.train()

    model.save_pretrained(os.path.join(main.out_dir,'model'))
    trainer.save_model(os.path.join(main.out_dir,'model_trainer')) 

    process_log_history(sweep_args,trainer.state.log_history,main.out_dir)
    
    final_preds(out_dir=main.out_dir,
                args={**vars(model_args), **vars(tune_args), **vars(sweep_args)},
                heldout_icd=d_heldout_icd,
                heldout_expert=d_heldout_expert)
    

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArgs,TuneArgs,SweepArgs))
    model_args, tune_args, sweep_args = parser.parse_args_into_dataclasses()

    main(model_args, tune_args, sweep_args)
