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
    EarlyStoppingCallback, IntervalStrategy,
)
import tokenizers
from tokenizers import (
    decoders, models, normalizers, pre_tokenizers, processors,
    trainers,Tokenizer,AddedToken, ByteLevelBPETokenizer,
)
from tokenizers.models import BPE
from tokenizers.decoders import BPEDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, BertPreTokenizer
from tokenizers.normalizers import (
    BertNormalizer, NFD, StripAccents, Replace, Strip,
)
from tokenizers.processors import TemplateProcessing, RobertaProcessing

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
os.environ['MASTER_PORT'] = '1111'

s1 = 1234 
seq_len = 4096
update_vocab_len = int(50000) #int(5e4)
min_freq = int(2) #10
no_punc = False

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
    
dat = read_data(os.path.join(data_dir,tbl_fn))

token_dir = os.path.join(out_dir,'token')
out_pretrain_dir = os.path.join(out_dir,'pretrain')
out_finetune_dir = os.path.join(out_dir,'finetune')

d_train = Dataset.from_dict(dat['train'])

def get_training_corpus():
    return (
        d_train[i : i + 1000]["text"]
        for i in range(0, len(d_train), 1000)
    )

training_corpus = get_training_corpus()

mod = 'yikuan8/Clinical-Longformer'
words = ['nissen','bidmc','beth','israel','deaconess','brigham','dimock','spaulding','bidmc',
         'arbor','shore','plymouth','carney','baptist','auburn','lawrence','cambridge',
         'haldol','seroquel','aoxtwo','aoxone','aoxthree','aoxzero']

tokenizer = AutoTokenizer.from_pretrained(mod,fast=True)
print('\n\nTesting words prior to training.')
for w in words:
    in_out = tokenizer.convert_ids_to_tokens(tokenizer.encode(w))
    print('%s: %s' % (w,in_out))
print('\nVocab length %s.' % len(tokenizer.vocab))
    
#tokenizer.normalizer = normalizers.Sequence([NFD(), BertNormalizer(), Strip(), StripAccents()])
tokenizer_update = tokenizer.train_new_from_iterator(training_corpus,
                                                     vocab_size=update_vocab_len, 
                                                     min_frequency=min_freq,
                                                     show_progress=True)

#tokenizer.add_tokens(list(tokenizer_update.vocab))

print('\n\nTesting words after training.')
for w in words:
    #in_out = tokenizer.convert_ids_to_tokens(tokenizer.encode(w))
    tokened = tokenizer.tokenize(w)
    print('%s: %s' % (w,tokened))
print('\nVocab length %s.' % len(tokenizer.vocab))

print('\n\nTesting words after training.')
for w in words:
    #in_out = tokenizer_update.convert_ids_to_tokens(tokenizer_update.encode(w))
    tokened = tokenizer_update.tokenize(w)
    print('%s: %s' % (w,tokened))
print('\nVocab length %s.' % len(tokenizer_update.vocab))

new_tokens = list(set(tokenizer_update.vocab.keys()) - set(tokenizer.vocab.keys()))
tokenizer.add_tokens(new_tokens)

print('\n\nTesting words after merging.')
for w in words:
    #in_out = tokenizer_update.convert_ids_to_tokens(tokenizer_update.encode(w))
    tokened = tokenizer.tokenize(w)
    print('%s: %s' % (w,tokened))
print('\nVocab length %s.' % len(tokenizer.vocab))

tokenizer_update.save_pretrained(token_dir)



