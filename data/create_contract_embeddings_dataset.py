import time
import random
import json
import pickle

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from clauserec_utils import get_embeddings_for_clause_set


def create_contract_embeddings_dataset(contract_clauses, filename, tokenizer, model, 
                                       batch_size=128, device='cpu'):
    
    contract_dicts = list()
    for i, (contract, label_clauses) in enumerate(contract_clauses.items()):
        clauses = list()
        labelset = set()
        for labels, clause in label_clauses:
            clauses.append(clause)
            for label in labels:
                labelset.add(label)
        
        avg_sent_mean, avg_sent_cls, avg_clause_mean, \
        avg_clause_cls = get_embeddings_for_clause_set(clauses, tokenizer, model, 
                                                       batch_size, device)
        
        contract_dict = dict()
        contract_dict['contract'] = contract
        contract_dict['n_clauses'] = len(clauses)
        contract_dict['labelset'] = list(labelset)
        contract_dict['avg_sent_mean'] = avg_sent_mean.cpu()
        contract_dict['avg_sent_cls'] = avg_sent_cls.cpu()
        contract_dict['avg_clause_mean'] = avg_clause_mean.cpu()
        contract_dict['avg_clause_cls'] = avg_clause_cls.cpu()
        contract_dicts.append(contract_dict)
        
        if (i+1) % 100 == 0:
            print(i+1, 'done, last contract:', contract)
        
    print('\nSaving contract embeddings .... ')
    with open(filename, 'wb') as f:
        pickle.dump(contract_dicts, f)
    print('Done saving file', filename)
        
    return


def print_stored_contract_embeddings(filename):
    with open(filename, 'rb') as f:
        contract_dicts = pickle.load(f)

    num_contracts = len(contract_dicts)
    print('No. of contracts:', num_contracts)
    print('Randomly sampling 10 for check', end='\n\n')
    sampled_cdicts = random.sample(range(num_contracts), 10)

    for sample_ind in sampled_cdicts:
        cdict = contract_dicts[sample_ind]
        print(cdict['contract'])
        print('labelset:', cdict['labelset'])
        print('No. of clauses:', cdict['n_clauses'])
        print('avg_sent_mean:', cdict['avg_sent_mean'].shape)
        print('avg_sent_cls:', cdict['avg_sent_cls'].shape)
        print('avg_clause_mean:', cdict['avg_clause_mean'].shape)
        print('avg_clause_cls:', cdict['avg_clause_cls'].shape, end='\n\n')
        
    return


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

with open('contract-clauses.json', 'r') as f:
    contract_clauses = json.load(f)
    
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

filename = 'contract-embeddings-robertabase.pkl'
create_contract_embeddings_dataset(contract_clauses, filename, tokenizer, model, batch_size=128, device=DEVICE)

print('Dataset creation complete, time taken: ', time.time() - start)

print_stored_contract_embeddings(filename)
