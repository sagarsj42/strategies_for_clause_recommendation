import re
import math

import nltk
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast


def clean_clause(clause, lowercase=True, normalize_all_caps=True):
    if not isinstance(clause, str):
        raise ValueError('Input clause is not a string.')
    
    cleaned = re.sub(r'[^A-Za-z0-9\s/()_\-&.,;"\':$%]', '', clause)
    cleaned = re.sub(r'__+', r'__', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    if lowercase:
        cleaned = cleaned.lower()
        
    if cleaned.isupper() and normalize_all_caps:
        cleaned = '. '.join(sent.capitalize() for sent in cleaned.split('. '))
        
    cleaned = cleaned.strip()
    
    return cleaned


def generate_square_subsequent_mask(sz: int=512, device='cpu') -> Tensor:
    return torch.triu(torch.ones((sz, sz), device=device) * float('-inf'), diagonal=1)


def get_wordpiece_tokenizer(max_length=512):
    tokenizer_file = 'clauserec-tokenizer-wordpiece.json'
    unk = '[UNK]'
    bos = '[BOS]'
    eos = '[EOS]'
    pad = '[PAD]'
    msk = '[MSK]'

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, model_max_length=max_length, 
        bos_token=bos, eos_token=eos, unk_token=unk, sep_token=eos, pad_token=pad, cls_token=bos, mask_token=msk)

    return fast_tokenizer


def average_embeddings_by_attention(embeds, mask):
    div1 = mask.sum(dim=1).view(-1, 1, 1)
    div2 = embeds.shape[0]
    att_enc = embeds * mask.unsqueeze(-1)
    avg_embed = (att_enc / div1).sum(dim=1).unsqueeze(1).sum(dim=0) / div2
    
    return avg_embed


def average_cls_embeddings(embeds):
    div = embeds.shape[0]
    avg_embed = embeds[:, 0, :].unsqueeze(1).sum(dim=0) / div
    
    return avg_embed


def get_embeddings_for_clause_set(clauses, tokenizer, model, batch_size=128, device='cpu'):
    n_clauses = len(clauses)
    if n_clauses > 1000:
        print('Calculating sent-wise embeddings:', n_clauses, 'steps')
    
    avg_sent_mean = torch.zeros((1, 768)).to(device)
    avg_sent_cls = torch.zeros((1, 768)).to(device)
    for clause in clauses:
        sents = nltk.sent_tokenize(clause)
        sents_tok = tokenizer(sents, truncation=True, padding=True, return_tensors='pt', 
                              return_attention_mask=True, return_token_type_ids=False).to(device)
        sents_emb = model(**sents_tok).last_hidden_state

        clause_embed_mean = average_embeddings_by_attention(sents_emb, sents_tok.attention_mask)
        clause_embed_cls = average_cls_embeddings(sents_emb)

        avg_sent_mean = torch.add(clause_embed_mean, avg_sent_mean)
        avg_sent_cls = torch.add(clause_embed_cls, avg_sent_cls)
    avg_sent_mean /= n_clauses
    avg_sent_cls /= n_clauses
    
    steps = math.ceil(n_clauses / batch_size)
    if steps > 10:
        print('Calculating clause-wise embeddings:', steps, 'steps')
    
    avg_clause_mean = torch.zeros((1, 768)).to(device)
    avg_clause_cls = torch.zeros((1, 768)).to(device)
    for i in range(1, steps+1):
        batch = clauses[(i-1)*batch_size: i*batch_size]
        batch_tok = tokenizer(batch, truncation=True, padding=True, return_tensors='pt', 
                              return_attention_mask=True, return_token_type_ids=False).to(device)
        batch_embed = model(**batch_tok).last_hidden_state

        mask = batch_tok.attention_mask
        div = mask.sum(dim=1).view(-1, 1, 1)
        batch_attn_enc = batch_embed * mask.unsqueeze(-1)
        batch_sum_mean = (batch_attn_enc / div).sum(dim=1).unsqueeze(1).sum(dim=0)
        
        batch_sum_cls = batch_embed[:, 0, :].sum(dim=0).unsqueeze(0)

        avg_clause_mean = torch.add(batch_sum_mean, avg_clause_mean)
        avg_clause_cls = torch.add(batch_sum_cls, avg_clause_cls)
    avg_clause_mean /= n_clauses
    avg_clause_cls /= n_clauses
    
    return avg_sent_mean, avg_sent_cls, avg_clause_mean, avg_clause_cls


def get_contract_representation(clauses, tokenizer, model, avg_sents=True, use_cls=False, device='cpu'):
    if avg_sents:
        contr_emb = torch.zeros((1, 768)).to(device)
        nclauses = len(clauses)
        
        for clause in clauses:
            sents = nltk.sent_tokenize(clause)
            sents_tok = tokenizer(sents, truncation=True, padding=True, return_token_type_ids=False, 
                                  return_attention_mask=True, return_tensors='pt').to(device)
            sents_emb = model(**sents_tok).last_hidden_state
            
            if use_cls:
                clause_emb = average_cls_embeddings(sents_emb)
            else:
                clause_emb = average_embeddings_by_attention(sents_emb, sents_tok['attention_mask'])
                
            contr_emb = torch.add(clause_emb, contr_emb)
        contr_emb /= float(nclauses)
    else:
        clauses_tok = tokenizer(clauses, truncation=True, padding=True, return_tensors='pt', 
                              return_token_type_ids=False, return_attention_mask=True).to(device)
        clauses_emb = model(**clauses_tok).last_hidden_state
        
        if use_cls:
            contr_emb = average_cls_embeddings(clauses_emb)
        else:
            contr_emb = average_embeddings_by_attention(clauses_emb, clauses_tok['attention_mask'])
            
    return contr_emb
