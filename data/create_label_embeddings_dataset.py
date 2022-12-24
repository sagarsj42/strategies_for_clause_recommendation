import time
import json
import pickle

import torch
from torch import nn

from transformers import AutoTokenizer, AutoModel

from clauserec_utils import get_embeddings_for_clause_set


def create_label_embeddings_dataset(label_clauses, filename, tokenizer, model, batch_size, device):
    label_dicts = dict()
    for label, clauses in label_clauses.items():
        print('Taking embeddings for', label)
        
        avg_sent_mean, avg_sent_cls, avg_clause_mean, \
        avg_clause_cls = get_embeddings_for_clause_set(clauses, tokenizer, model, 
                                                       batch_size, device)
        
        label_dict = dict()
        label_dict['n_clauses'] = len(clauses)
        label_dict['avg_sent_mean'] = avg_sent_mean
        label_dict['avg_sent_cls'] = avg_sent_cls
        label_dict['avg_clause_mean'] = avg_clause_mean
        label_dict['avg_clause_cls'] = avg_clause_cls
        label_dicts[label] = label_dict
        
    print('\nSaving label embeddings .... ')
    with open(filename, 'wb') as f:
        pickle.dump(label_dicts, f)
    print('Done saving file', filename)
        
    return


def print_stored_label_embeddings(filename):
    with open(filename, 'rb') as f:
        label_dicts = pickle.load(f)

    print('No. of labels:', len(label_dicts), end='\n\n')

    for label, ldict in label_dicts.items():
        print(label)
        print('No. of clauses:', ldict['n_clauses'])
        print('avg_sent_mean:', ldict['avg_sent_mean'].shape)
        print('avg_sent_cls:', ldict['avg_sent_cls'].shape)
        print('avg_clause_mean:', ldict['avg_clause_mean'].shape)
        print('avg_clause_cls:', ldict['avg_clause_cls'].shape, end='\n\n')
        
    return


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

start = time.time()
with open('label-clauses.json', 'r') as f:
    label_clauses = json.load(f)
print('Time to load label_clauses.json:', time.time() - start)

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
tokenizer.model_max_length = 512
model = AutoModel.from_pretrained('roberta-base').to(DEVICE)
model = nn.DataParallel(model)

for p in model.named_parameters():
    print(p[0], end=' ')
    p[1].requires_grad = False
    print(p[1].requires_grad)
    
start = time.time()
print('Starting dataset creation ....')

filename = 'label-embeddings-robertabase.pkl'
create_label_embeddings_dataset(label_clauses, filename, tokenizer, model, batch_size=128, device=DEVICE)

print('Dataset creation complete, time taken: ', time.time() - start)

print_stored_label_embeddings(filename)
