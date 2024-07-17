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

import wandb

from trl import SFTTrainer

from transformers import (
    AutoTokenizer, DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    AutoModelForSequenceClassification,
    AutoConfig, TrainingArguments,
    AutoModelForMaskedLM,BitsAndBytesConfig,
    HfArgumentParser,
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

from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, replace_lora_weights_loftq

@dataclass
class SweepArgs:
    """
    Arguements for sweeping through Longformer params.
    """
    
    threshold: Optional[int] = field(default=70)
    fraction: Optional[int] = field(default=100 )
    label: Optional[str] = field(default='only')
    pipeline: Optional[int] = field(default=1)
    
    def __post_init__(self):
        valid_thresholds = [60,70,80,90]
        valid_fractions = [20,35,50,75,100]
        valid_labels = ['icd','full','only','pseudo']
        valid_pipelines = [1,2,3]
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
    testing: Optional[bool] = field(default=False)
    num_labels: Optional[int] = field(default=2)
    log_steps: Optional[int] = field(default=None)
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
    mod: Optional[int] = field(default=8) 
    f_subset_data: Optional[float] = field(default=None)
    bf16: Optional[bool] = field(default=True)
    mod_load_in_8bit: Optional[bool] = field(default=False)
    mod_torch_dtype: Optional[str] = field(default='bfloat16')
    bnb_load_in_4bit: Optional[bool] = field(default=True)
    bnb_4bit_quant_type: Optional[str] = field(default='nf4')
    bnb_4bit_use_double_quant: Optional[bool] = field(default=True)
    bnb_4bit_compute_dtype: Optional[str] = field(default='bfloat16')
    bnb_4bit_quant_storage_dtype: Optional[str] = field(default='bfloat16')
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    lara_bias: Optional[str] = field(default='none')
    lora_target_modules: Optional[str] = field(default='q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj')

    def __post_init__(self):
        self.quant_compute_dtype = getattr(torch, self.bnb_4bit_quant_storage_dtype)
        self.quant_storage_dtype = getattr(torch, self.bnb_4bit_quant_storage_dtype)
        self.mod_torch_dtype = getattr(torch, self.mod_torch_dtype)
        valid_mods= [8,70]
        self.mod = 'meta-llama/Meta-Llama-3-8B' if self.mod == 8 else 'meta-llama/Meta-Llama-3-70B'

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

class wSFTTrainer(SFTTrainer):
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
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\nCuda device: {device}")
    
    if device != 'cpu':
        gc.collect()
        torch.cuda.empty_cache()
    
    if model_args.n_steps_testing is not None:
        os.environ['WANDB_DISABLED'] = 'true'

    if model_args.seed is None:
        set_seed(random.randint(1,99999))
    else:
        set_seed(model_args.seed)
    
    main.folder_fn = get_folder_fn(model_args,sweep_args)
    
    return device, main.folder_fn

def get_components(sweep_args,model_args,quantization_config,out_dir,folder_fn):
            
        train_dir = os.path.join(out_dir,'finetune',folder_fn) # needs a unique folder since labels/strf

        model = AutoModelForSequenceClassification.from_pretrained(model_args.mod,
                                                                   #quantization_config=quantization_config,
                                                                   trust_remote_code=True,
                                                                   load_in_8bit=model_args.mod_load_in_8bit,
                                                                   attn_implementation='flash_attention_2',
                                                                   torch_dtype=model_args.mod_torch_dtype,
                                                                   num_labels=model_args.num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_args.mod,max_length=model_args.seq_len)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        out_dir = os.path.join(train_dir,'final_model_finetune') # location to output mod, pl1

        return model, tokenizer, out_dir

def get_data(model_args,sweep_args,tune_args):
    dat = read_data(os.path.join('/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/data',
                                 model_args.input_table),
                    th=sweep_args.threshold,fr=sweep_args.fraction,exp=sweep_args.label,
                   limit=120000)

    class_weights = None
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

    d_heldout_icd = Dataset.from_dict(dat['heldout_icd'])
    d_heldout_expert = Dataset.from_dict(dat['heldout_expert'])
    d_heldout_expert_filtered = Dataset.from_dict(dat['heldout_expert_filtered'])

    return d_train, d_val, d_heldout_icd, d_heldout_expert, d_heldout_expert_filtered, class_weights

def get_cfgs(model_args, sweep_args):
    bnb_config = BitsAndBytesConfig( 
        load_in_4bit = model_args.bnb_load_in_4bit,
        bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=model_args.quant_compute_dtype,
        bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
        bnb_4bit_quant_storage=model_args.bnb_4bit_quant_storage_dtype,
    )

    peft_config = LoraConfig(
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        r=model_args.lora_r,
        bias=model_args.lara_bias,
        target_modules=model_args.lora_target_modules,
    )
    
    peft_config.task_type = 'SEQ_CLS'
    
    return bnb_config, peft_config

def get_folder_fn(model_args,sweep_args):
    folder_fn = 'fit_' + sweep_args.label
    if sweep_args.pipeline == 'pseudo':
        folder_fn += '_th' + th + '_fr' + fr
    folder_fn += '_pl' + str(sweep_args.pipeline) 
    if model_args.folder_suffix is not None:
        folder_fn += '_' + model_args.folder_suffix 

    return(folder_fn)

def main(model_args, tune_args, sweep_args):
    print('\nRun args:')
    print_vars({**vars(model_args), **vars(tune_args), **vars(sweep_args)})
    
    device, main.folder_fn = init(model_args, sweep_args)
    main.out_dir = os.path.join(model_args.work_dir,'out')

    d_train, d_val, d_heldout_icd, d_heldout_expert, d_heldout_expert_filtered, class_weights = get_data(model_args,sweep_args,tune_args)
    bnb_config, peft_config = get_cfgs(model_args, sweep_args)
    model, tokenizer, main.out_dir = get_components(sweep_args,model_args,bnb_config,main.out_dir,main.folder_fn)
    
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
        n_steps = int(len(d_train) * model_args.n_train_epochs / model_args.n_batch / model_args.n_grad_accum)
        n_warmup = int(n_steps * model_args.warmup_ratio)
        n_epochs = model_args.n_train_epochs
        log_steps = n_steps // model_args.f_log_steps
        save_strategy = 'steps'
        save_steps = log_steps * model_args.save_multiplier
    print(f"\nRun will be over {n_steps} training steps, {log_steps} evaluation steps, {n_warmup} warmup steps, and {n_epochs} epochs.")

    print('\nTokenizing training data.')
    d_train = d_train.map(tokenize_dataset,batched=True,num_proc=model_args.n_cores,remove_columns=['text'])
    print('Tokenizing validation data.')
    d_val = d_val.map(tokenize_dataset,batched=True,num_proc=model_args.n_cores,remove_columns=['text'])
    

    print('Tokenizing testing data.')
    d_heldout_icd = d_heldout_icd.map(tokenize_dataset,batched=True,num_proc=model_args.n_cores,remove_columns=['text'])
    d_heldout_expert = d_heldout_expert.map(tokenize_dataset,batched=True,num_proc=model_args.n_cores,remove_columns=['text'])
    d_heldout_expert_filtered = d_heldout_expert_filtered.map(tokenize_dataset,batched=True,
                                                              num_proc=model_args.n_cores,remove_columns=['text'])

    if tune_args.upsample:
        print('Upsampling training data.')
        d_train = balance_data(d_train,cores=model_args.n_cores)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    if tune_args.group_by_len:
        data_collator.padding = 'longest'
    else:
        data_collator.padding = 'max_length'
        data_collator.max_length = model_args.seq_len


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
        
        max_steps = n_steps if model_args.testing else -1,

        num_train_epochs = n_epochs,
        learning_rate = model_args.lr, 
        weight_decay = model_args.w_decay,  
        warmup_steps = n_warmup,
        lr_scheduler_type = 'constant_with_warmup',
        
        load_best_model_at_end = True,
        metric_for_best_model = 'b_accuracy',
        greater_is_better = True,
        label_smoothing_factor = model_args.label_smoothing,

        dataloader_num_workers = model_args.n_cores, 
        dataloader_persistent_workers = True,
        dataloader_pin_memory = True,

        group_by_length = tune_args.group_by_len,

        bf16 = model_args.bf16,

        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {'use_reentrant':False},
        gradient_accumulation_steps = model_args.n_grad_accum,
        eval_accumulation_steps = model_args.n_grad_accum_eval,

        per_device_train_batch_size = model_args.n_batch, 
        per_device_eval_batch_size = model_args.n_batch_eval, 
    )
    
    if model_args.n_steps_testing is None:
        training_args.report_to = 'wandb'
        training_args.run_name = main.folder_fn
        
    print(f"\nTraining groups: 0={d_train['labels'].count(0)}, 1={d_train['labels'].count(1)}")
    print(f"Validation groups: 0={d_val['labels'].count(0)}, 1={d_val['labels'].count(1)}")
    print(f"Heldout icd groups: 0={d_heldout_icd['labels'].count(0)}, 1={d_heldout_icd['labels'].count(1)}")
    print(f"Heldout expert groups: 0={d_heldout_expert['labels'].count(0)}, 1={d_heldout_expert['labels'].count(1)}")

    print('\nTraining args:')
    print_vars(vars(training_args))
          
    if tune_args.class_weights:
        print('\nUsing weighted loss function.')
        class cTrainer(wSFTTrainer):
            pass
        cTrainer.load_class_weights(class_weights)
    else:
        class cTrainer(SFTTrainer):
            pass
            
    trainer = cTrainer(
        model = model,
        args = training_args,
        train_dataset = d_train,
        eval_dataset = d_val,
        peft_config=peft_config,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics,
        data_collator = data_collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5,early_stopping_threshold=0.005)],
    )
    
    trainer.accelerator.print(f"Model: {trainer.model}")
    trainer.model.print_trainable_parameters()
    
    if getattr(trainer.accelerator.state, 'fsdp_plugin', None):
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
   
    with open(os.path.join(main.out_dir,'eval_results.dat'), 'w') as f:
        print('acc\tb_acc\tf1\tauc\tprec\trec\tbatch_len\tpred_pod\ttrue_pos\n',file=f)
            
    trainer.train()

    trainer.save_model(os.path.join(main.out_dir,'model_trainer')) 

    process_log_history(sweep_args,trainer.state.log_history,main.out_dir)
    
    final_preds(out_dir=main.out_dir,
                args={**vars(model_args), **vars(tune_args), **vars(sweep_args)},
                **{'heldout_icd': d_heldout_icd, 'heldout_expert': d_heldout_expert})


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArgs,TuneArgs,SweepArgs))
    model_args, tune_args, sweep_args = parser.parse_args_into_dataclasses()
    
    os.environ["WANDB_PROJECT"]= 'lf_llama'                     
    os.environ["WANDB_LOG_MODEL"] = 'false'

    main(model_args, tune_args, sweep_args)
