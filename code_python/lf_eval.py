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

# import custom functions
from lf_functions import *

os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(random.randint(1000, 9999))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('\n\nNumber of devices: %s.\n \
    Device set to %s.\n' % (torch.cuda.device_count(),device))

update_vocab_len = int(5e4) # size of new tokenizer
min_freq = int(2) # filter word freq

tbl_fn = 'tbl_to_python_expertupdate_chunked.csv.gz'

work_dir = '/home/swolosz1/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer'
data_dir = os.path.join(work_dir,'data')
out_dir = os.path.join(work_dir,'out')

token_dir = os.path.join(out_dir,'token')

dat = read_data(os.path.join(data_dir,tbl_fn),st=False,train_on_expert=False,finetuning=False)

# just using training data for tokenization
# no val samples which are saved strictly for validation during ft
# and no training expert labeled samples
d_train = Dataset.from_dict(dat['train'])

# create sample iterator
def get_training_corpus():
    return (
        d_train[i : i + 1000]['text']
        for i in range(0, len(d_train), 1000)
    )

training_corpus = get_training_corpus()

mod = 'yikuan8/Clinical-Longformer' # repo clinical longformer
tokenizer = AutoTokenizer.from_pretrained(mod,fast=True)

# words for testing tokenizer
words = ['nissen','bidmc','beth','israel','deaconess','brigham','dimock','spaulding','bidmc',
         'arbor','shore','plymouth','carney','baptist','auburn','lawrence','cambridge',
         'haldol','seroquel','aoxtwo','aoxone','aoxthree','aoxzero']

print('\n\nTesting words prior to training.')
for w in words:
    in_out = tokenizer.convert_ids_to_tokens(tokenizer.encode(w))
    print('%s: %s' % (w,in_out))
print('\nVocab length %s.' % len(tokenizer.vocab))

# training new tokenzizer from new data based on original parameterization
tokenizer_update = tokenizer.train_new_from_iterator(training_corpus,
                                                     vocab_size=update_vocab_len, 
                                                     min_frequency=min_freq,
                                                     show_progress=True)

print('\n\nTesting words after training.')
for w in words:
    tokened = tokenizer.tokenize(w)
    print('%s: %s' % (w,tokened))
print('\nVocab length %s.' % len(tokenizer.vocab))

print('\n\nTesting words after training.')
for w in words:
    tokened = tokenizer_update.tokenize(w)
    print('%s: %s' % (w,tokened))
print('\nVocab length %s.' % len(tokenizer_update.vocab))

# obtain new tokens not in repo tokenizer then add them to repo tokenizer
new_tokens = list(set(tokenizer_update.vocab.keys()) - set(tokenizer.vocab.keys()))
tokenizer.add_tokens(new_tokens)

print('\n\nTesting words after merging.')
for w in words:
    tokened = tokenizer.tokenize(w)
    print('%s: %s' % (w,tokened))
print('\nVocab length %s.' % len(tokenizer.vocab))

tokenizer_update.save_pretrained(token_dir)



