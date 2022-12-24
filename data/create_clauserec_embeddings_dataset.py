import os
import time
import random
import pickle

import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertModel

from clauserec_utils import clean_clause, get_contract_representation


def create_clauserec_embeddings_dataset(train_split, dev_split, dir_name, tokenizer, model, device='cpu'):
    for split in ('train', 'dev'):
        print('Starting', split)
        os.makedirs(os.path.join(dir_name, split), exist_ok=True)
        if split == 'train':
            split_df = train_split
        else:
            split_df = dev_split

        for i in range(split_df.shape[0]):
            row_id = split_df.index[i]
            row = split_df.loc[row_id]
            instance = dict()

            instance['id'] = row_id
            clauses = row['contract_clauses']
            instance['contract_clauses'] = clauses

            clauses = [clean_clause(clause) for clause in clauses]
            instance['avg_sent_mean'] = get_contract_representation(clauses, tokenizer, model, 
                                                                    avg_sents=True, use_cls=False, device=device)
            instance['avg_sent_cls'] = get_contract_representation(clauses, tokenizer, model, 
                                                                   avg_sents=True, use_cls=True, device=device)
            instance['avg_clause_mean'] = get_contract_representation(clauses, tokenizer, model, 
                                                                      avg_sents=False, use_cls=False, device=device)
            instance['avg_clause_cls'] = get_contract_representation(clauses, tokenizer, model, 
                                                                     avg_sents=False, use_cls=True, device=device)

            instance['label'] = row['label']
            instance['rec_clause'] = row['rec_clause']
            instance['other_output_ids'] = row['other_output_ids']
            instance['contract'] = row['contract']

            with open(os.path.join(dir_name, split, str(row_id) + '.pkl'), 'wb') as f:
                pickle.dump(instance, f)
                
            if i % 1000 == 0 and i != 0:
                print('\tStored file no.', i)
                
        print(split, 'complete', '\n\n')
                
    return


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', torch.cuda.device_count(), 'GPUs')

tokenizer = BertTokenizer.from_pretrained('./pretrain-bert-contr-mlm')
model = BertModel.from_pretrained('./pretrain-bert-contr-mlm')
model = nn.DataParallel(model).to(DEVICE)

print('-'*70)
for p in model.named_parameters():
    print(p[0])
    p[1].requires_grad = False
print('-'*70)

start = time.time()

train_split = pd.read_parquet('clauserec-train.parquet', engine='fastparquet')
dev_split = pd.read_parquet('clauserec-dev.parquet', engine='fastparquet')

print(train_split.info(), '\n')
print(dev_split.info(), '\n')

print('Time to load data:', time.time() - start)
print('Starting dataset creation ....')

start = time.time()
dir_name = 'clauserec-mlm'
create_clauserec_embeddings_dataset(train_split, dev_split, dir_name, 
                                    tokenizer=tokenizer, model=model, device=DEVICE)
print('Dataset creation over!')
print('Time taken to create:', time.time() - start)

for split in ('train', 'dev'):
    files = os.listdir(os.path.join(dir_name, split))
    nfiles = len(files)
    print(f'No. of {split} files: {nfiles}')
    
    sample_no = random.randint(0, nfiles-1)
    file = files[sample_no]
    with open(os.path.join(dir_name, split, file), 'rb') as f:
        instance = pickle.load(f)
        
    print(f'Randomly sampled file: {file}')
    print('id: {id}, label: {label}, contract: {contract} \nother_output_ids: {other_output_ids}'.format(**instance))
    print('No. of clauses: {0}'.format(len(instance['contract_clauses'])))
    print('avg_sent_mean - shape: {0}, first 10: {1}'.format(instance['avg_sent_mean'].shape, 
                                                             instance['avg_sent_mean'][0, :10]))
    print('avg_sent_cls - shape: {0}, first 10: {1}'.format(instance['avg_sent_cls'].shape, 
                                                            instance['avg_sent_cls'][0, :10]))
    print('avg_clause_mean - shape: {0}, first 10: {1}'.format(instance['avg_clause_mean'].shape, 
                                                               instance['avg_clause_mean'][0, :10]))
    print('avg_clause_cls - shape: {0}, first 10: {1}'.format(instance['avg_clause_cls'].shape, 
                                                              instance['avg_clause_cls'][0, :10]))
    print()
