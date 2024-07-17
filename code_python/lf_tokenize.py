import os
import shutil

from datasets import Dataset

from transformers import AutoTokenizer

from lf_functions import read_data

update_vocab_len = int(5e4) # size of new tokenizer
min_freq = int(2) # filter word freq

tbl_fn = 'tbl.csv.gz'

work_dir = '/home/swolosz1/shared/anesthesia/wolosomething/delirium/cleanrun_01/longformer'
data_dir = os.path.join(work_dir,'data')
out_dir = os.path.join(work_dir,'out')
token_dir = os.path.join(out_dir,'token')

# create dir if doesnt exist
if os.path.exists(token_dir):
    shutil.rmtree(token_dir)
os.makedirs(token_dir)

print('Reading data.')
dat = read_data(os.path.join(data_dir,tbl_fn),exp='pretrain')

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



