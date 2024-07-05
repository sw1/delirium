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
import shutil
import tempfile

import matplotlib.pyplot as plt
from pynvml import *
from plotnine import *

import torch
from torch.nn import CrossEntropyLoss

import ray
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import logging

import wandb
from ray.air.integrations.wandb import WandbLoggerCallback

from transformers import (
    AutoTokenizer, DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    LongformerForSequenceClassification,
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

from typing import Optional, Callable
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
    eval_steps: Optional[int] = field(default=1000)
    n_batch_eval: Optional[int] = field(default=64)
    n_train_epochs: Optional[int] = field(default=4)
    lr: Optional[float] = field(default=8e-06 )
    warmup_ratio: Optional[float] = field(default=0.07)
    n_warmup: Optional[int] = field(default=500)
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
    overwrite_prompt: Optional[bool] = field(default=True)
    test_run: Optional[bool] = field(default=False)
    n_trials: Optional[int] = field(default=1)
    asha_max_epochs: Optional[float] = field(default=5), 
    asha_gp: Optional[float] = field(default=0.5),
    asha_rf: Optional[int] = field(default=2),
    sweep_path: Optional[str] = field(default='/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/out/sweep')
    
    def __post_init__(self):
        
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
    f1 = evaluate.load('f1').compute(predictions=preds, references=labels)['f1']
    bacc = balanced_accuracy_score(y_true=labels,y_pred=preds)

    return {'accuracy': acc, 'b_accuracy': bacc, 
            'f1': f1, 'auc': auc, 'precision': prec, 'recall': rec, 
            'batch_length': len(preds),'pred_positive': sum(preds), 'true_positive': sum(labels)}

def get_folder_fn(model_args,sweep_args):
    if sweep_args.train_method == 'finetune':   
        folder_fn = 'fit_' + sweep_args.label
        if sweep_args.pipeline == 'pseudo':
            folder_fn += '_th' + th + '_fr' + fr
        folder_fn += '_pl' + str(sweep_args.pipeline) 
        if model_args.folder_suffix is not None:
            folder_fn += '_' + model_args.folder_suffix 
    else:
        folder_fn = None

    return(folder_fn)

def init():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        
    if model_args.seed is None:
        set_seed(random.randint(1,99999))
    elif model_args.seed == 0:
        pass
    else:
        set_seed(model_args.seed)
        
def update_tokenizer(tokenizer,model,token_dir):
    tokenizer_update = AutoTokenizer.from_pretrained(token_dir,fast=True)
    new_tokens = list(set(tokenizer_update.vocab.keys()) - set(tokenizer.vocab.keys()))
    tokenizer.add_tokens(new_tokens)
    dim1 = str(model.get_input_embeddings())
    model.resize_token_embeddings(len(tokenizer))
    dim2 = str(model.get_input_embeddings())

    return tokenizer, model

def get_components(model_args, sweep_args,out_dir, folder_fn):

        if sweep_args.train_method == 'finetune':   
            
            train_dir = os.path.join(out_dir,'finetune',folder_fn) # needs a unique folder since labels/strf
            
            if sweep_args.pipeline == 1: # finetune with repo model
                mod = 'yikuan8/Clinical-Longformer' # repo clinical lf
                conf = AutoConfig.from_pretrained(mod,num_labels=model_args.num_labels) 
                conf.hidden_dropout_prob=model_args.do_hidden
                conf.classifier_dropout=model_args.do_class
                model = LongformerForSequenceClassification.from_pretrained(mod,config=conf)
                tokenizer = AutoTokenizer.from_pretrained(mod,use_fast=True,max_length=model_args.seq_len)
                out_dir = os.path.join(train_dir,'final_model_finetune') # location to output mod, pl1
            elif sweep_args.pipeline == 2: # finetune with pretrained model
                mod = os.path.join(model_pretrain,'model') # pretrained mod
                conf = AutoConfig.from_pretrained(mod,num_labels=model_args.num_labels)
                conf.hidden_dropout_prob=model_args.do_hidden
                conf.classifier_dropout=model_args.do_class
                model = LongformerForSequenceClassification.from_pretrained(mod,config=conf)
                tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                                          use_fast=True,max_length=model_args.num_labels)
                out_dir = os.path.join(train_dir,'final_model_pretrain_finetune') # location to output mod, pl2
            elif sweep_args.pipeline == 3: # finetune with pretrained model that used custom tokenizer
                mod = os.path.join(model_token_pretrain,'model') # pretrained mod using custom tok
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
            
            if pl == 1: # pretrain with repo model
                out_dir = os.path.join(train_dir,'model_pretrain') # loc for pretrained mod
            if pl == 2: # pretrain with custom tokenizer
                # obtain new tokens not in repo tokenizer then add them to repo tokenizer
                tokenizer, model = update_tokenizer(tokenizer,model,token_dir)
                out_dir = os.path.join(train_dir,'model_token_pretrain') # loc for custom tok and pretrained mod
                
        return model, tokenizer, conf, out_dir
    
def get_data(model_args, tune_args, sweep_args):
    dat = read_data(os.path.join(model_args.work_dir,'data',model_args.input_table),
                    th=sweep_args.threshold,fr=sweep_args.fraction,exp=sweep_args.label)

    class_weights = None
    if sweep_args.train_method == 'finetune':
        if tune_args.class_weights:
            class_weights = get_class_weights(dat['train'], model_args.num_labels)

    if tune_args.filter_keywords:
        for i in range(len(dat['train']['text'])):
            note = dat['train']['text'][i]
            note = note.replace('delirium','').replace('encephalopathy','')
            dat['train']['text'][i] = note

    d_train = Dataset.from_dict(dat['train'])

    if tune_args.test_run:
        d_train = d_train.train_test_split(test_size=0.99,shuffle=True)['train'] 

    d_val = Dataset.from_dict(dat['val'])

    if sweep_args.train_method == 'finetune':
        d_heldout_icd = Dataset.from_dict(dat['heldout_icd'])
        d_heldout_expert = Dataset.from_dict(dat['heldout_expert'])
    else:
        d_heldout_icd = d_heldout_expert = class_weights = None

    return d_train, d_val, d_heldout_icd, d_heldout_expert, class_weights
        
def train_fxn(config):    
    os.environ['MASTER_PORT'] = str(random.randint(1000, 9999))
    
    init()
    
    n_cores = ray._private.utils.get_num_cpus() // torch.cuda.device_count()
    
    folder_fn = get_folder_fn(config['model_args'],config['sweep_args'])
    config['tune_args'].sweep_path = os.path.join(config['tune_args'].sweep_path,folder_fn)
    
    config['tune_args'].filter_keywords = config['filter_keywords']
    d_train, d_val, d_heldout_icd, d_heldout_expert, class_weights = get_data(config['model_args'], config['tune_args'], config['sweep_args'])
    model, tokenizer, conf, _ = get_components(config['model_args'], sweep_args, os.path.join(config['model_args'].work_dir,'out'), folder_fn)

    n_steps = len(d_train)*config['tune_args'].asha_max_epochs/config['n_batch']
    n_steps = math.floor(n_steps/2 if config['n_batch'] == 64 else n_steps)
    eval_steps = round(n_steps * model_args.f_log_steps)
    eval_steps = 100 if eval_steps > 100 else eval_steps
    
    def tokenize_dataset(data):
        data['text'] = [line for line in data['text'] if len(line) > 0 and not line.isspace()]

        return tokenizer(data['text'],padding=config['tune_args'].pad_max_len,truncation=True,
                         max_length=config['model_args'].seq_len,return_special_tokens_mask=True)

    d_train = d_train.map(tokenize_dataset,batched=True,num_proc=n_cores,remove_columns=['text'])
    d_val = d_val.map(tokenize_dataset,batched=True,num_proc=n_cores,remove_columns=['text'])
    
    if config['sweep_args'].train_method == 'finetune':
        if config['tune_args'].upsample:
            d_train = balance_data(d_train,cores=n_cores)
            
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        if config['tune_args'].group_by_len:
            data_collator.padding = 'longest'
        else:
            data_collator.padding = 'max_length'
            data_collator.max_length = config['model_args'].seq_len
            
    elif config['sweep_args'].train_method == 'pretrain':
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        
    training_args = TrainingArguments(
        disable_tqdm = True,

        do_train = True,
        do_eval = True,

        save_steps = 0,
        logging_dir = './logs',
        
        output_dir = '.',
        
        report_to='wandb',
        run_name=tune.Trainable().trial_id,

        save_strategy = 'steps',

        evaluation_strategy = 'steps',
        eval_steps = config['model_args'].eval_steps, 
        
        load_best_model_at_end = False,
        
        num_train_epochs = 1 if config['tune_args'].test_run else tune_args.asha_max_epochs,
        learning_rate = config['lr'], #tune
        weight_decay = config['w_decay'], #tune
        
        max_steps = 1 if config['tune_args'].test_run else -1,
        
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        warmup_steps = 0 if config['tune_args'].test_run else config['model_args'].n_warmup,

        auto_find_batch_size = False, 
        dataloader_num_workers = n_cores, 
        dataloader_persistent_workers = True,
        dataloader_pin_memory = True,

        tf32 = True,

        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {'use_reentrant':False},
        gradient_accumulation_steps = math.ceil(config['n_batch']/16),
        eval_accumulation_steps = config['model_args'].n_grad_accum_eval,

        per_device_train_batch_size = config['n_batch'], #tune
        per_device_eval_batch_size = config['model_args'].n_batch_eval, 
    )
    
    if config['sweep_args'].train_method == 'finetune':
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = 'b_accuracy'
        training_args.greater_is_better = True
        training_args.label_smoothing_factor = config['lab_smooth'] #config['model_args'].label_smoothing #tune
        
    training_args.group_by_length = config['tune_args'].group_by_len
          
    if config['tune_args'].class_weights and config['sweep_args'].train_method == 'finetune':
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
    )
    
    if config['sweep_args'].train_method == 'finetune':
        trainer.compute_metrics = compute_metrics
        if config['tune_args'].use_collator:
            trainer.data_collator = data_collator

    elif config['sweep_args'].train_method == 'pretrain':
        trainer.data_collator = data_collator
 
    trainer.train()

    eval_results = trainer.evaluate()
    eval_b_accuracy = eval_results.get('eval_b_accuracy')

    print(f"\nEvalulation output:\n{eval_results}")
    
    metrics = {'eval_b_accuracy': eval_b_accuracy, 
               'filter_keywords': config['filter_keywords'],
               'lr': config['lr'],
               'w_decay': config['w_decay'],
               'n_batch': config['n_batch'],
               'lab_smooth': config['lab_smooth']}
    
    if ray.train.get_context().get_world_rank() == 0:
        print(f"\nMetrics:\n{metrics}")
    
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        trainer.save_model(temp_checkpoint_dir)
        trainer.save_state()
        trainer.save_metrics('eval', eval_results)
        tokenizer.save_pretrained(temp_checkpoint_dir)
        session.report(metrics,checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir))


def tune_transformer(model_args, tune_args, sweep_args):
    
    folder_fn = get_folder_fn(model_args, sweep_args)
   
    tune_config = {
        'model_args': model_args,
        'tune_args': tune_args,
        'sweep_args': sweep_args,

        'filter_keywords': tune.choice([True,False]),
        'lr': tune.loguniform(2e-7, 2e-5),
        'w_decay': tune.loguniform(5e-4, 5e-1),
        'n_batch': tune.choice([8, 16, 32, 64]),
        'lab_smooth': tune.loguniform(3e-10, 3e-1),
    }
 
    scheduler = tune.schedulers.ASHAScheduler(
        time_attr='epoch',
        max_t=tune_args.asha_max_epochs, 
        grace_period=tune_args.asha_gp, 
        reduction_factor=tune_args.asha_rf,
    )

    reporter = CLIReporter(
        parameter_columns={'filter_keywords': 'fkw',
                           'lr': 'lr',
                           'w_decay': 'wd',
                           'n_batch': 'nb',
                           'lab_smooth': 'ls'
                          },
        metric_columns={'epoch': 'e',
                        'eval_loss': 'los',
                        'eval_b_accuracy': 'bac'
        },
        max_report_frequency=1000,
        print_intermediate_tables=True,
    )

    sweep = tune.run(
        tune.with_parameters(train_fxn),
        metric='eval_b_accuracy',
        mode='max',
        resources_per_trial={'cpu': ray._private.utils.get_num_cpus() // torch.cuda.device_count(), 
                             'gpu': 1},
        config=tune_config,
        num_samples=torch.cuda.device_count() if tune_args.test_run else tune_args.n_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        name='sweep_' + folder_fn,
        storage_path=os.path.join(tune_args.sweep_path,folder_fn,'tuning_results'),  
        stop={'training_iteration': 2} if tune_args.test_run else None,
        keep_checkpoints_num=3, 
        callbacks=[WandbLoggerCallback(
            project='sweep_lf',
            log_config=True,
            reinit=True,
            allow_val_change=True
        )]
    )
    
    
    best_trial = sweep.get_best_trial(metric='eval_b_accuracy',mode='max')
    print(f"\nBest trial config: {best_trial.config}")
    
    best_trial_path = best_trial.checkpoint.path
    shutil.copytree(best_trial_path, os.path.join(tune_args.sweep_path,'best_trial'))
    
    best_checkpoint_path = sweep.get_best_checkpoint(best_trial,metric='b_eval_accuracy',mode='max').checkpoint.path
    shutil.copytree(best_checkpoint_path, os.path.join(tune_args.sweep_path,'best_trial_checkpoint'))
    
if __name__ == '__main__':
    parser = HfArgumentParser((ModelArgs,TuneArgs,SweepArgs))
    model_args, tune_args, sweep_args = parser.parse_args_into_dataclasses()
    
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    
    os.environ["TRANSFORMERS_VERBOSITY"] = 'error'
    os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = '0'
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = '1'
    
    os.environ["RAY_memory_usage_threshold"] = '0.98'
    #os.environ["RAY_memory_monitor_refresh_ms"] = '0'
    #os.environ["RAY_USE_MULTIPROCESSING_CPU_COUNT"] = '1'
    
    if tune_args.test_run:
        os.environ['WANDB_DISABLED'] = 'true'
    else:
        os.environ["WANDB_PROJECT"] = 'sweep_lf'
        os.environ["WANDB_LOG_MODEL"] = 'false'
        
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    ray.init(ignore_reinit_error=True)
    
    tune_transformer(model_args, tune_args, sweep_args)

    ray.shutdown()     
