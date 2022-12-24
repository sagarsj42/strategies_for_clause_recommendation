import os

os.chdir('working_dir')
os.environ['TORCH_HOME'] = 'working_dir'
os.environ['TRANSFORMERS_CACHE'] = 'working_dir'

import argparse
import re
import time
import random
import pickle
import json
from collections import defaultdict
from functools import partial

import numpy as np

import torch
from torch.utils.data import Subset, DataLoader

from rouge_score import rouge_scorer
from nltk.translate import bleu_score
from thefuzz import fuzz
from sklearn.model_selection import StratifiedShuffleSplit

from model.clause_decoder import ClauseDecoder, ClauseDecoderWithLinear
from data.clauserec_dataset import *
from data.clauserec_collators import *
from eval.output_processor import get_model_outputs_with_tokenizer
from utils.indexing import construct_index
from utils.clauserec_utils import get_wordpiece_tokenizer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

parser = argparse.ArgumentParser()
parser.add_argument('--DATA_PATH', type=str, required=True, help='Path to the unzipped serialized dataset for the bert-based encoder')
parser.add_argument('--MODEL_PATH', type=str, required=True, help='Path to the outputs of the experiment to be evaluated')
parser.add_argument('--N_EPOCHS', type=int, required=True, help='No. of saved epochs in the experiment')
parser.add_argument('--BATCH_SIZE', type=int, required=True, help='Batch size to be used for taking model outputs')
parser.add_argument('--RETRIEVE_K', type=int, required=False, help='Retrieval size from index querying')


args = parser.parse_args()

print('Arguments specified:', args.DATA_PATH, args.LABEL_EMBED_FILE, args.MODEL_PATH, args.N_EPOCHS, args.BATCH_SIZE)

DATA_PATH = args.DATA_PATH
MODEL_PATH = args.MODEL_PATH
N_EPOCHS = args.N_EPOCHS
BATCH_SIZE = args.BATCH_SIZE
RETRIEVE_K = args.RETRIEVE_K

# DATA_PATH = 'clauserec-lbbase'
# MODEL_PATH = 'clausedecoder-lbbase-avg_clause_mean-only_contract'
# N_EPOCHS = 50
# BATCH_SIZE = 64
# RETRIEVE_K = 6

if 'mlm' in MODEL_PATH:
    IDS_FILE = 'clauserec-mlm-3split-ids.json'
elif 'lbbase' in MODEL_PATH:
    IDS_FILE = 'clauserec-lbbase-3split-ids.json'
elif 'lbcontracts' in MODEL_PATH:
    IDS_FILE = 'clauserec-lbcontracts-3split-ids.json'

if 'mlm' in MODEL_PATH:
    LABEL_EMBED_FILE = 'label-embeddings-bertmlm.pkl'
elif 'lbbase' in MODEL_PATH:
    LABEL_EMBED_FILE = 'label-embeddings-lbbase.pkl'
elif 'lbcontracts' in MODEL_PATH:
    LABEL_EMBED_FILE = 'label-embeddings-lbcontracts.pkl'

if 'mlm' in MODEL_PATH:
    CONTRACTS_EMBED_FILE = 'contract-embeddings-bertmlm.pkl'
elif 'lbbase' in MODEL_PATH:
    CONTRACTS_EMBED_FILE = 'contract-embeddings-lbbase.pkl'
elif 'lbcontracts' in MODEL_PATH:
    CONTRACTS_EMBED_FILE = 'contract-embeddings-lbcontracts.pkl'

if 'mlm' in MODEL_PATH:
    IDS_FILE = 'clauserec-mlm-3split-ids.json'
elif 'lbbase' in MODEL_PATH:
    IDS_FILE = 'clauserec-lbbase-3split-ids.json'
elif 'lbcontracts' in MODEL_PATH:
    IDS_FILE = 'clauserec-lbcontracts-3split-ids.json'

if 'avg_sent_mean' in MODEL_PATH:
    EMBED_TYPE = 'avg_sent_mean'
elif 'avg_sent_cls' in MODEL_PATH:
    EMBED_TYPE = 'avg_sent_cls'
elif 'avg_clause_mean' in MODEL_PATH:
    EMBED_TYPE = 'avg_clause_mean'
elif 'avg_clause_cls' in MODEL_PATH:
    EMBED_TYPE = 'avg_clause_cls'

if 'contract_and_label' in MODEL_PATH:
    STRATEGY = 'contract_and_label'
elif 'only_contract' in MODEL_PATH:
    STRATEGY = 'only_contract'
elif 'contract_label_fullsim_contract' in MODEL_PATH:
    STRATEGY = 'contract_label_fullsim_contract'
elif 'contract_label_sim_contr_clauses' in MODEL_PATH:
    STRATEGY = 'contract_label_sim_contr_clauses'
elif 'only_contract_fullsim_contract' in MODEL_PATH:
    STRATEGY = 'only_contract_fullsim_contract'

if 'mlm' in DATA_PATH:
    SIM_CLAUSE_EMB_FILE = 'simclause-embeddings-bertmlm.pkl'
elif 'lbbase' in DATA_PATH:
    SIM_CLAUSE_EMB_FILE = 'simclause-embeddings-lbbase.pkl'
elif 'lbcontracts' in DATA_PATH:
    SIM_CLAUSE_EMB_FILE = 'simclause-embeddings-lbcontracts.pkl'

with open(LABEL_EMBED_FILE, 'rb') as f:
        label_embeds = pickle.load(f)

with open(CONTRACTS_EMBED_FILE, 'rb') as f:
    contract_embeds = pickle.load(f)

with open(SIM_CLAUSE_EMB_FILE, 'rb') as f:
    simclause_emb = pickle.load(f)

INDEXING_INFO = {
    'indexing_strategy': 'faiss.IndexHNSWFlat',
    'd': 768,
    'm': 64,
    'ef_construction': 128,
    'ef_search': 64,
    'retrieve_k': RETRIEVE_K
}
index = construct_index(contract_embeds, EMBED_TYPE, INDEXING_INFO['d'], INDEXING_INFO['m'], 
    INDEXING_INFO['ef_construction'], INDEXING_INFO['ef_search'])

tokenizer = get_wordpiece_tokenizer()

if STRATEGY == 'only_contract':
    collate_fn = partial(collate_only_contract_with_tokenizer, tokenizer=tokenizer)
elif STRATEGY == 'contract_and_label':
    collate_fn = partial(collate_with_label_and_tokenizer, embed_type=EMBED_TYPE, 
        label_embeds=label_embeds, tokenizer=tokenizer)
elif STRATEGY == 'contract_label_fullsim_contract':
    collate_fn = partial(collate_with_label_and_full_similar_contracts, embed_type=EMBED_TYPE, 
        label_embeds=label_embeds, contract_embeds=contract_embeds, index=index, hnsw_k=RETRIEVE_K, 
        tokenizer=tokenizer)
elif STRATEGY == 'contract_label_sim_contr_clauses':
    collate_fn = partial(collate_with_label_and_serialized_select_clause_rep_from_similar_contracts, embed_type=EMBED_TYPE,
        label_embeds=label_embeds, contract_embeds=contract_embeds, simclause_emb=simclause_emb,
        index=index, hnsw_k=RETRIEVE_K, tokenizer=tokenizer)
elif STRATEGY == 'only_contract_fullsim_contract':
    collate_fn = partial(collate_only_contract_and_full_similar_contracts, embed_type=EMBED_TYPE, 
        contract_embeds=contract_embeds, index=index, hnsw_k=RETRIEVE_K, tokenizer=tokenizer)

with open('label-clauses.json', 'r') as f:
    label_clauses = json.load(f)

print('DATA_PATH:', DATA_PATH, 'LABEL_EMBED_FILE:', LABEL_EMBED_FILE, 'CONTRACTS_EMBED_FILE:', CONTRACTS_EMBED_FILE)
print('MODEL_PATH:', MODEL_PATH, 'EMBED_TYPE:', EMBED_TYPE, 'STRATEGY:', STRATEGY)
print('N_EPOCHS:', N_EPOCHS, 'BATCH_SIZE:', BATCH_SIZE)


def has_matching_char(string, char='\''):
    count = 0
    for c in string:
        if c == char:
            count += 1
    return count % 2 == 0


def apply_output_regex(string):
    string = re.sub(r' \.', r'.', string)
    string = re.sub(r' ,', r',', string)
    string = re.sub(r' ;', r';', string)
    string = re.sub(r'\( ', r'(', string)
    string = re.sub(r' \)', r')', string)
    
    if has_matching_char(string, char='\''):
        string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    if has_matching_char(string, char='"'):
        string = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', string)
    
    return string


def get_label_outputs(outputs, split, data_path):
    label_outputs = defaultdict(list)

    for sample_id, output, actual in outputs:
        with open(os.path.join(data_path, split, str(sample_id)+'.pkl'), 'rb') as f:
            sample = pickle.load(f)

        label = sample['label']
#         actual = sample['rec_clause'].lower().strip()
        
        output = clean_clause(output)
        actual = clean_clause(actual)
        output = apply_output_regex(output)
        actual = apply_output_regex(actual)
        label_outputs[label].append((output, actual))
        
    return label_outputs


def calculate_rouge(label_outputs, use_stemmer=False):
    rouge_sc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
    # Takes in args as rouge_scorer.score(target, prediction)
    
    label_rouges = dict()
    overall_rouge = {
        'rouge1': np.zeros(3),
        'rouge2': np.zeros(3),
        'rougeL': np.zeros(3)
    }
    n_samples = 0

    for label in label_outputs:
        n_label_samples = float(len(label_outputs[label]))
        rouge_dict = {
            'rouge1': np.zeros(3),
            'rouge2': np.zeros(3),
            'rougeL': np.zeros(3)
        }

        for output, actual in label_outputs[label]:
            sample_rouge = rouge_sc.score(actual, output)

            rouge_dict['rouge1'] += sample_rouge['rouge1'][:3]
            rouge_dict['rouge2'] += sample_rouge['rouge2'][:3]
            rouge_dict['rougeL'] += sample_rouge['rougeL'][:3]

            overall_rouge['rouge1'] += sample_rouge['rouge1'][:3]
            overall_rouge['rouge2'] += sample_rouge['rouge2'][:3]
            overall_rouge['rougeL'] += sample_rouge['rougeL'][:3]

        rouge_dict['rouge1'] = list(rouge_dict['rouge1'] / float(n_label_samples))
        rouge_dict['rouge2'] = list(rouge_dict['rouge2'] / float(n_label_samples))
        rouge_dict['rougeL'] = list(rouge_dict['rougeL'] / float(n_label_samples))

        label_rouges[label] = rouge_dict
        n_samples += n_label_samples

    overall_rouge['rouge1'] = list(overall_rouge['rouge1'] / float(n_samples))
    overall_rouge['rouge2'] = list(overall_rouge['rouge2'] / float(n_samples))
    overall_rouge['rougeL'] = list(overall_rouge['rougeL'] / float(n_samples))
    
    return label_rouges, overall_rouge


def calculate_bleu(label_outputs):
    all_outputs = list()
    all_actuals = list()
    all_sent_bleus = list()
    label_bleus = dict()
    smoothing_function = bleu_score.SmoothingFunction().method3

    for label in label_outputs:
        outputs = [output.split() for output, _ in label_outputs[label]]
        actuals = [[actual.split()] for _, actual in label_outputs[label]]

        label_sent_bleus = list()
        for output, actual in zip(outputs, actuals):
            sent_bl = bleu_score.sentence_bleu(actual, output, smoothing_function=smoothing_function)
            label_sent_bleus.append(sent_bl)

        label_bleus[label] = {
            'corpus_bleu': bleu_score.corpus_bleu(actuals, outputs, 
                smoothing_function=smoothing_function, weights=(0.25, 0.25, 0.25, 0.25)),
            'bleu1': bleu_score.corpus_bleu(actuals, outputs, 
                smoothing_function=smoothing_function, weights=(1, 0, 0, 0)),
            'bleu2': bleu_score.corpus_bleu(actuals, outputs, 
                smoothing_function=smoothing_function, weights=(0, 1, 0, 0)),
            'sentence_bleu': float(np.array(label_sent_bleus).mean())
        }

        all_outputs.extend(outputs)
        all_actuals.extend(actuals)
        all_sent_bleus.extend(label_sent_bleus)

    overall_bleu = {
        'corpus_bleu': bleu_score.corpus_bleu(all_actuals, all_outputs, smoothing_function=smoothing_function, 
            weights=(0.25, 0.25, 0.25, 0.25)),
        'bleu1': bleu_score.corpus_bleu(all_actuals, all_outputs, smoothing_function=smoothing_function, 
            weights=(1, 0, 0, 0)),
        'bleu2': bleu_score.corpus_bleu(all_actuals, all_outputs, smoothing_function=smoothing_function, 
            weights=(0, 1, 0, 0)),
        'sentence_bleu': float(np.array(all_sent_bleus).mean())
    }
    
    return label_bleus, overall_bleu


def calculate_jaccard_similarity(string_a, string_b):
    set_a = set(string_a.split())
    set_b = set(string_b.split())
    
    intersect = set_a.intersection(set_b)
    union = set_a.union(set_b)
    
    sim_score = len(intersect) / len(union)
    
    return sim_score


def calculate_string_match_stats(label_outputs, label_clauses, top_labels):
    string_matches = dict()

    for label in top_labels:
        label_matches = defaultdict(int)
        outputs = label_outputs[label]
        n_outputs = len(outputs)
        avg_max_score = 0
        avg_max_jaccard = 0
        
        if n_outputs == 0:
            continue
            
        for output_clause, _ in outputs:
            max_score = 0
            max_jaccard = 0
            for actual_clause in label_clauses[label]:
                actual_clause = apply_output_regex(actual_clause.lower())
                score = fuzz.ratio(actual_clause, output_clause)
                jaccard = calculate_jaccard_similarity(actual_clause, output_clause) * 100
                
                if score > max_score:
                    max_score = score
                    
                if jaccard > max_jaccard:
                    max_jaccard = jaccard
                    
                if max_score == 100 and max_jaccard == 100:
                    break
                    
            avg_max_score += max_score
            avg_max_jaccard += max_jaccard

            if max_score == 100:
                label_matches['100'] += 1
            elif max_score >= 95:
                label_matches['95-100'] += 1
            elif max_score >= 90:
                label_matches['90-95'] += 1
            elif max_score >= 85:
                label_matches['85-90'] += 1
            elif max_score >= 80:
                label_matches['80-85'] += 1
            elif max_score >= 75:
                label_matches['75-80'] += 1
            else:
                label_matches['<75'] += 1
                
        avg_max_score /= n_outputs
        avg_max_jaccard /= n_outputs
        
        label_matches['avg_max_score'] = avg_max_score
        label_matches['avg_max_jaccard'] = avg_max_jaccard

        string_matches[label] = label_matches
        
    return string_matches


def calculate_metrics_with_tokenizer(checkpoints, model_path, train_dataloader, dev_dataloader, 
    batch_size, tokenizer, label_clauses, data_path, device='cpu'):
    
    results = dict()

    for checkpoint in checkpoints:
        print('Starting checkpoint:', checkpoint)
        checkpoint_results = dict()

        save_dict = torch.load(os.path.join(model_path, checkpoint))
        
        # Use ClauseDecoderWithLinear for experiments involving similar contracts. Else, use ClauseDecoder
        model = ClauseDecoder(**save_dict['model_args'])
        
        model.load_state_dict(save_dict['model_state_dict'])
        model.to(device)
        model.eval()

        checkpoint_results['meta'] = {
            'experiment_name': save_dict['experiment_name'],
            'checkpoint': checkpoint,
        }
        for k in save_dict:
            if k != 'model_state_dict' and k != 'optimizer_state_dict':
                checkpoint_results['meta'][k] = save_dict[k]

        # split = 'train'
        # print('Split:', split)
        # train_start = time.time()
        # train_ids = [train_dataset[i][0] for i in range(len(train_dataset))]
        # train_outputs, train_actuals = get_model_outputs_with_tokenizer(train_dataloader, tokenizer, model, 
        #     batch_size=batch_size, device=device)
        
        # train_out_set = [(idx, output, actual) for idx, output, actual in 
        #                  zip(train_ids, train_outputs, train_actuals)]

        # train_outfile_name = os.path.join(model_path, 
        #                                   f'{checkpoint[:-4]}-{split}-{len(train_dataset)}-outputs.pkl')
        # with open(train_outfile_name, 'wb') as f:
        #     pickle.dump(train_out_set, f)

        # print('Outputs saved in:', train_outfile_name)

        # train_label_outputs = get_label_outputs(train_out_set, split, data_path)
        # train_label_rouges, train_overall_rouge = calculate_rouge(train_label_outputs, use_stemmer=True)
        # train_label_bleus, train_overall_bleu = calculate_bleu(train_label_outputs)
        # train_time = time.time() - train_start
        
        # print('Time taken for outputs + eval:', train_time)

        # checkpoint_results['train'] = {
        #     'label_rouges': train_label_rouges,
        #     'overall_rouge': train_overall_rouge,
        #     'label_bleus': train_label_bleus,
        #     'overall_bleu': train_overall_bleu,
        #     'time_taken': train_time
        # }

        # split = 'dev'
        # print('Split:', split)
        # dev_start = time.time()
        # dev_ids = [dev_dataset[i][0] for i in range(len(dev_dataset))]
        # dev_outputs, dev_actuals = get_model_outputs_with_tokenizer(dev_dataloader, tokenizer, model, 
        #     batch_size=batch_size, device=device)
        
        # dev_out_set = [(idx, output, actual) for idx, output, actual in 
        #                  zip(dev_ids, dev_outputs, dev_actuals)]
        # dev_outfile_name = os.path.join(model_path, 
        #                                 f'{checkpoint[:-4]}-{split}-{len(dev_dataset)}-outputs.pkl')
        # with open(dev_outfile_name, 'wb') as f:
        #     pickle.dump(dev_out_set, f)

        # print('Outputs saved in:', dev_outfile_name)

        # dev_label_outputs = get_label_outputs(dev_out_set, split, data_path)
        # dev_label_rouges, dev_overall_rouge = calculate_rouge(dev_label_outputs, use_stemmer=True)
        # dev_label_bleus, dev_overall_bleu = calculate_bleu(dev_label_outputs)
        
        # dev_time = time.time() - dev_start
        
        # print('Time taken for outputs + eval:', dev_time)

        # checkpoint_results['dev'] = {
        #     'label_rouges': dev_label_rouges,
        #     'overall_rouge': dev_overall_rouge,
        #     'label_bleus': dev_label_bleus,
        #     'overall_bleu': dev_overall_bleu,
        #     'time_taken': dev_time
        # }

        split = 'test'
        print('Split:', split)
        test_start = time.time()
        test_ids = [test_dataset[i][0] for i in range(len(test_dataset))]
        test_outputs, test_actuals = get_model_outputs_with_tokenizer(test_dataloader, tokenizer, model, 
            batch_size=batch_size, device=device)
        
        test_out_set = [(idx, output, actual) for idx, output, actual in 
            zip(test_ids, test_outputs, test_actuals)]
        test_outfile_name = os.path.join(model_path, 
            f'{checkpoint[:-4]}-{split}-{len(test_dataset)}-outputs.pkl')
        with open(test_outfile_name, 'wb') as f:
            pickle.dump(test_out_set, f)

        print('Outputs saved in:', test_outfile_name)

        test_label_outputs = get_label_outputs(test_out_set, 'dev', data_path)
        test_label_rouges, test_overall_rouge = calculate_rouge(test_label_outputs, use_stemmer=True)
        test_label_bleus, test_overall_bleu = calculate_bleu(test_label_outputs)
        
        test_time = time.time() - test_start
        
        print('Time taken for outputs + eval:', test_time)

        checkpoint_results['test'] = {
            'label_rouges': test_label_rouges,
            'overall_rouge': test_overall_rouge,
            'label_bleus': test_label_bleus,
            'overall_bleu': test_overall_bleu,
            'time_taken': test_time
        }

        results[checkpoint] = checkpoint_results
        
        print(checkpoint, 'complete')

    return results


#ids, _ = filter_clauserec_exclude_all_but_one(DATA_PATH)

# with open('clauserec-filtered-ids.json', 'w') as f:
#    json.dump(ids,f)

## Code to generate eval ids file

with open(IDS_FILE, 'r') as f:
    ids = json.load(f)

train_dataset = ClauserecDataset(DATA_PATH, ids, 'train', EMBED_TYPE)
dev_dataset = ClauserecDataset(DATA_PATH, ids, 'dev', EMBED_TYPE)
test_dataset = ClauserecDataset(DATA_PATH, ids, 'dev', EMBED_TYPE)

train_dataset = Subset(train_dataset, list(range(10)))
dev_dataset = Subset(dev_dataset, list(range(10)))
test_dataset = Subset(test_dataset, list(range(1100)))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# train_labels = list()
# dev_labels = list()

# for i in range(len(train_dataset)):
#    train_labels.append(train_dataset[i][2])

# for i in range(len(dev_dataset)):
#     dev_labels.append(dev_dataset[i][2])

# splitter = StratifiedShuffleSplit(n_splits=1, test_size=2100, random_state=43419)
# sampled_ids = dict()

# for _, select_index in splitter.split(train_dataset.ids, y=train_labels):
#    sampled_ids['train'] = [train_dataset.ids[si] for si in select_index]
    
# for _, select_index in splitter.split(dev_dataset.ids, y=dev_labels):
#    sampled_ids['dev'] = [dev_dataset.ids[si] for si in select_index]
    
# with open('clauserec-eval-ids.json', 'w') as f:
#    json.dump(sampled_ids, f)

checkpoints = list()
# checkpoints = ['epoch-' + str(i) + '.pth' for i in range(1, N_EPOCHS+1)]
checkpoints.append('best.pth')

# results = calculate_metrics(checkpoints, model_path=MODEL_PATH, data_path=DATA_PATH, 
#     batch_size=BATCH_SIZE, collate_fn=collate_fn, label_clauses=label_clauses, 
#     device=DEVICE)

results = calculate_metrics_with_tokenizer(checkpoints, model_path=MODEL_PATH, train_dataloader=train_dataloader, 
    dev_dataloader=dev_dataloader, batch_size=BATCH_SIZE, tokenizer=tokenizer, label_clauses=label_clauses, 
    data_path=DATA_PATH, device=DEVICE)

print(results)

results_file = f'{MODEL_PATH}.results.json'
with open(results_file, 'w') as f:
    json.dump(results, f)
    
print(f'All the metric results for {MODEL_PATH} are saved in {results_file}.')
