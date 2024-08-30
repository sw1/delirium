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

from transformers import (
    AutoTokenizer, DataCollatorWithPadding,
    Trainer,
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
from datasets import load_dataset, Dataset, DatasetDict

from sklearn.utils import compute_class_weight
from sklearn.metrics import (precision_recall_fscore_support,
                             balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split

from typing import Optional
from dataclasses import dataclass, field
import torch.nn.functional as F

from peft.utils.other import fsdp_auto_wrap_policy

# import custom functions
from lf_functions import (
    read_data, softmax, compute_eval_metrics, sigfigs,
    balance_data, get_class_weights, check_and_save_params,
    pull_results, print_vars, process_log_history, final_preds,
)

from accelerate import PartialState, Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, replace_lora_weights_loftq

class wTrainer(Trainer):
    class_weights = None
    
    @classmethod
    def load_class_weights(cls, class_weights):
        cls.class_weights = class_weights
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(self.class_weights,dtype=torch.float32).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels').long()
        
        outputs = model(**inputs)
        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    global trend_bacc

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

    trend_bacc_0 = max(trend_bacc)
    if (len(trend_bacc) >= 10): # delay
        trend_bacc.pop(0)
    trend_bacc_1 = bacc
    if trend_bacc_1 - trend_bacc_0 > 1.0e-04:
        update_trend_bacc = trend_bacc_1 # ensures position of 'best model' based on trend
    else:
        update_trend_bacc = trend_bacc_0 - 1.0e-06
    trend_bacc.append(update_trend_bacc)
    
    return {'accuracy': acc, 'b_accuracy': bacc, 'trend_b_accuracy': update_trend_bacc,
            'f1': f1, 'auc': auc, 'precision': prec, 'recall': rec, 
            'prop_pred_positive': ppp, 'prop_true_positive': ptp,
            'prop_diff': prop_diff, 'score': score}

def chunk_sample(sample, chunk_size=8192):
    chunks = []
    
    # Calculate dynamic overlap
    num_chunks = (len(sample['input_ids']) + chunk_size - 1) // chunk_size
    
    start = 0
    if num_chunks > 1:
        overlap = max(1, (num_chunks * chunk_size - len(sample['input_ids'])) // (num_chunks - 1))
        while start + chunk_size <= len(sample['input_ids']):
            chunks.append({
                'id': sample['id'],
                'labels': sample['labels'],
                'input_ids': sample['input_ids'][start:start + chunk_size],
                'attention_mask': sample['attention_mask'][start:start + chunk_size],
                'special_tokens_mask': sample['special_tokens_mask'][start:start + chunk_size],
            })
            start += chunk_size - overlap
        
    # Handle the last chunk if it doesn't fit perfectly
    if start < len(sample['input_ids']):
        chunks.append({
            'id': sample['id'],
            'labels': sample['labels'],
            'input_ids': sample['input_ids'][-chunk_size:],
            'attention_mask': sample['attention_mask'][-chunk_size:],
            'special_tokens_mask': sample['special_tokens_mask'][-chunk_size:],
        })
    
    return chunks

def chunk_dataset(dataset, chunk_size=8192):
    new_dataset = {'id': [], 'labels': [], 'input_ids': [], 'attention_mask': [], 'special_tokens_mask': []}
    
    for sample in dataset:
        chunks = chunk_sample(sample, chunk_size)
        for chunk in chunks:
            new_dataset['id'].append(chunk['id'])
            new_dataset['labels'].append(chunk['labels'])
            new_dataset['input_ids'].append(chunk['input_ids'])
            new_dataset['attention_mask'].append(chunk['attention_mask'])
            new_dataset['special_tokens_mask'].append(chunk['special_tokens_mask'])
            
    return Dataset.from_dict(new_dataset)

chunk_notes = True
class_weighting = 0.05
seq_len = 8192 
n_train_epochs = 2
n_eff_batch = 16
n_batch = 1
n_gpus = torch.cuda.device_count()
n_grad_accum = n_eff_batch // n_gpus
lr = 5e-5
w_decay = 0.1
filter_keywords = True
th = 90
fr = 100
lab = 'pseudo'
pl = 1
num_evals = 50

accelerator = Accelerator()

os.environ["WANDB_PROJECT"]= 'llama'

seed = 12341
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
set_seed(seed)
#enable_full_determinism(model_args.seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

# may help with gpu efficiency/speed
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

folder_name = ('fkw' + str(int(filter_keywords)) +
               '_th' + str(th) +
               '_fr' + str(fr) + 
               '_pl' + str(pl) +
               '_lr' + sigfigs(lr,1) +
               '_wd' + sigfigs(w_decay,1) + 
               '_nb' + str(n_eff_batch) + 
               '_ls' + sigfigs(0.0,1) +
               '_cw' + str(int(class_weighting * 100)) +
               '_lab' + lab +
               '_chunk' + str(int(chunk_notes))) 

work_dir = '/shared/anesthesia/wolosomething/delirium/cleanrun_01/llama'
out_dir = os.path.join(work_dir,'out',folder_name)

dat = read_data(os.path.join('/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer/data',
                             'tbl_allnotes_trimmedlen.csv.gz'),
                th=th,fr=fr,exp=lab)

if filter_keywords:
    print('\nFiltering keywords from training set used in expert labeling.')
    for i in range(len(dat['train']['text'])):
            note = dat['train']['text'][i]
            note = note.replace('delirium','').replace('encephalopathy','')
            dat['train']['text'][i] = note
                
class_weights = torch.as_tensor([1-class_weighting, class_weighting])
print(f"Class weights:\n0={class_weights[0]}\n1={class_weights[1]}") 
                
dat['heldout_expert_filtered'] = dat['heldout_expert'] 
for i in range(len(dat['heldout_expert_filtered']['text'])):
    note = dat['heldout_expert_filtered']['text'][i]
    note = note.replace('delirium','').replace('encephalopathy','')
    dat['heldout_expert_filtered']['text'][i] = note

d_train = Dataset.from_dict(dat['train'])
d_val = Dataset.from_dict(dat['val'])
d_heldout_icd = Dataset.from_dict(dat['heldout_icd'])
d_heldout_expert = Dataset.from_dict(dat['heldout_expert'])
d_heldout_expert_filtered = Dataset.from_dict(dat['heldout_expert_filtered'])

dataset = DatasetDict({
    'train': d_train,
    'val': d_val,
    'test_icd': d_heldout_icd,
    'test_expert': d_heldout_expert,
    'test_expert_filtered': d_heldout_expert_filtered
})

del d_train, d_val, d_heldout_icd, d_heldout_expert, d_heldout_expert_filtered

bnb_config = BitsAndBytesConfig( 
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,
)

peft_config = LoraConfig(
    lora_alpha=16, # larger values leads to more significant updates and increases risk of overfitting
    lora_dropout=0.05, 
    r=16, # rank of low rank adaptation, larger values are more expressive but more costly
    bias='none', 
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'], # # essentially the attention mechanism layers
    task_type = 'SEQ_CLS',
)

model = AutoModelForSequenceClassification.from_pretrained('meta-llama/Meta-Llama-3-8B',
                                                           quantization_config=bnb_config,
                                                           attn_implementation='flash_attention_2',
                                                           torch_dtype=torch.bfloat16,
                                                           num_labels=2)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B',max_length=seq_len,add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

#model = prepare_model_for_kbit_training(model)

for name, param in model.named_parameters():
    param.requires_grad = False

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})

model = get_peft_model(model, peft_config)

if chunk_notes:
    def tokenize_dataset(data):
        data['text'] = [line for line in data['text'] if len(line) > 0 and not line.isspace()]

        return tokenizer(data['text'],padding=False,truncation=False,
                         max_length=seq_len,return_special_tokens_mask=True)
else:
    def tokenize_dataset(data):
        data['text'] = [line for line in data['text'] if len(line) > 0 and not line.isspace()]

        return tokenizer(data['text'],padding=False,truncation=True,
                         max_length=seq_len,return_special_tokens_mask=True)    

print('\nTokenizing data.')
dataset = dataset.map(tokenize_dataset,batched=True,num_proc=16,remove_columns=['text'])

if chunk_notes:
    dataset = DatasetDict({split: chunk_dataset(dataset[split],chunk_size=seq_len) for split in dataset})

n_steps = int(len(dataset['train']) * n_train_epochs / n_batch // n_grad_accum)
n_warmup = int(n_steps * 0.1 // n_train_epochs)
n_epochs = n_train_epochs
log_steps = int(n_steps // num_evals)
save_strategy = 'steps'
save_steps = log_steps 
print(f"\nRun will be over {n_steps} training steps, {log_steps} evaluation steps, {n_warmup} warmup steps, and {n_epochs} epochs.")
    
dataset.set_format('torch')

if chunk_notes:
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding='max_length',max_length=seq_len)
else:
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding='longest')

print('Loading training args.')
training_args = TrainingArguments(
    disable_tqdm = False,

    do_train = True,
    do_eval = False,

    output_dir = out_dir,
    logging_dir = os.path.join(out_dir,'log'),
    overwrite_output_dir = True,
    logging_strategy = 'steps',
    logging_steps = log_steps, 

    save_strategy = 'no', # throws an error on saves due to fsdp

    evaluation_strategy = 'steps',
    eval_steps = log_steps, 

    max_steps = -1,

    num_train_epochs = n_epochs,
    learning_rate = lr,   
    weight_decay = w_decay,   
    warmup_steps = n_warmup,
    lr_scheduler_type = 'linear', 
    optim='adamw_torch',
    
    label_smoothing_factor = 0.0,

    dataloader_num_workers = 16, 
    dataloader_persistent_workers = True,
    dataloader_pin_memory = True,

    group_by_length = True,

    bf16 = True,
    
    seed = seed,
    data_seed = seed,
    
    report_to = 'wandb',
    run_name = 'llama_testing_decay',

    gradient_accumulation_steps = n_grad_accum,
    eval_accumulation_steps = n_grad_accum,

    per_device_train_batch_size = n_batch, 
    per_device_eval_batch_size = n_batch, 
)

if chunk_notes:
    training_args.group_by_length = False
       
print('\nTraining args:')
print_vars(vars(training_args))

if class_weighting != 0.5:
    print('\nUsing weighted loss function.')
    class cTrainer(wTrainer):
        pass
    cTrainer.load_class_weights(class_weights)
else:
    class cTrainer(Trainer):
        pass

print('Loading trainer.')
trainer = cTrainer(
    model = model,
    args = training_args,
    train_dataset = dataset['train'],
    eval_dataset = dataset['val'],
    tokenizer = tokenizer,
    compute_metrics = compute_metrics,
    data_collator = data_collator,
)

trainer.model.print_trainable_parameters()
    
fsdp_plugin = trainer.accelerator.state.fsdp_plugin
fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

trend_bacc = [0.0]
trainer.train()

if trainer.is_fsdp_enabled:
     trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')

print('\nSaving model.')
trainer.save_model(os.path.join(out_dir,'model_trainer')) 
