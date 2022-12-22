import torch

from clauserec_utils import clean_clause, generate_square_subsequent_mask
from indexing import get_similar_contract_reps, get_serialized_select_clause_rep_from_similar_contract_reps


def collate_only_contract_with_tokenizer(batch, tokenizer):
    ids = list()
    embeds_batch = list()
    rec_clauses = list()

    for idx, embedding, _, rec_clause, _ in batch:
        ids.append(int(idx))
        embeds_batch.append(embedding)
        rec_clauses.append(clean_clause(rec_clause))
        
    ids = torch.tensor(ids, dtype=torch.long)
    src = torch.cat(embeds_batch).unsqueeze(0)
    tgt = tokenizer(rec_clauses, truncation=True, padding=True, return_tensors='pt', 
        return_token_type_ids=False, return_attention_mask=False).input_ids.transpose(0, 1)
    tgt_input = tgt[:-1, :]
    tgt_mask = generate_square_subsequent_mask(tgt_input.shape[0])
    tgt_padding_mask = (tgt_input == tokenizer.pad_token_id).transpose(0, 1)
    
    return ids, src, tgt, tgt_mask, tgt_padding_mask


def collate_with_label_and_tokenizer(batch, embed_type, label_embeds, tokenizer):
    ids = list()
    embeds_batch = list()
    rec_clauses = list()

    for idx, embedding, label, rec_clause, _ in batch:
        ids.append(int(idx))
        clause_type_embed = label_embeds[label][embed_type]
        combined_embed = torch.add(embedding, clause_type_embed) / 2.0
        embeds_batch.append(combined_embed)
        rec_clauses.append(clean_clause(rec_clause))
        
    ids = torch.tensor(ids, dtype=torch.long)
    src = torch.cat(embeds_batch).unsqueeze(0)
    tgt = tokenizer(rec_clauses, truncation=True, padding=True, return_tensors='pt', 
        return_token_type_ids=False, return_attention_mask=False).input_ids.transpose(0, 1)
    tgt_input = tgt[:-1, :]
    tgt_mask = generate_square_subsequent_mask(tgt_input.shape[0])
    tgt_padding_mask = (tgt_input == tokenizer.pad_token_id).transpose(0, 1)
    
    return ids, src, tgt, tgt_mask, tgt_padding_mask


def collate_with_label_and_full_similar_contracts(batch, embed_type, label_embeds, contract_embeds, index, hnsw_k, tokenizer):
    ids = list()
    query_contract_embeds = list()
    embeds_batch = list()
    rec_clauses = list()
    contr_names = list()

    for idx, embedding, label, rec_clause, contr in batch:
        ids.append(int(idx))
        query_contract_embeds.append(embedding)
        clause_type_embed = label_embeds[label][embed_type]
        combined_embed = torch.add(embedding, clause_type_embed) / 2.0
        embeds_batch.append(combined_embed)
        rec_clauses.append(clean_clause(rec_clause))
        contr_names.append(contr)
        
    query_contract_embeds = torch.cat(query_contract_embeds)
    sim_contr_rep = get_similar_contract_reps(index, hnsw_k, contract_embeds, query_contract_embeds, contr_names, embed_type)
    
    ids = torch.tensor(ids, dtype=torch.long)
    input_contr_ct_rep = torch.cat(embeds_batch)
    src = torch.cat([input_contr_ct_rep, sim_contr_rep], dim=1).unsqueeze(0)
    tgt = tokenizer(rec_clauses, truncation=True, padding=True, return_tensors='pt', 
        return_token_type_ids=False, return_attention_mask=False).input_ids.transpose(0, 1)
    tgt_input = tgt[:-1, :]
    tgt_mask = generate_square_subsequent_mask(tgt_input.shape[0])
    tgt_padding_mask = (tgt_input == tokenizer.pad_token_id).transpose(0, 1)
    
    return ids, src, tgt, tgt_mask, tgt_padding_mask


def collate_with_label_and_serialized_select_clause_rep_from_similar_contracts(batch, embed_type, label_embeds, contract_embeds, 
    simclause_emb, index, hnsw_k, tokenizer):
    
    ids = list()
    query_contract_embeds = list()
    embeds_batch = list()
    batch_labels = list()
    rec_clauses = list()
    contr_names = list()

    for idx, embedding, label, rec_clause, contr in batch:
        ids.append(int(idx))
        query_contract_embeds.append(embedding)
        clause_type_embed = label_embeds[label][embed_type]
        combined_embed = torch.add(embedding, clause_type_embed) / 2.0
        embeds_batch.append(combined_embed)
        batch_labels.append(label)
        rec_clauses.append(clean_clause(rec_clause))
        contr_names.append(contr)
        
    query_contract_embeds = torch.cat(query_contract_embeds)
    sim_contr_rep = get_serialized_select_clause_rep_from_similar_contract_reps(index, hnsw_k, contract_embeds, query_contract_embeds, 
        contr_names, simclause_emb, batch_labels)
    
    ids = torch.tensor(ids, dtype=torch.long)
    input_contr_ct_rep = torch.cat(embeds_batch)
    src = torch.cat([input_contr_ct_rep, sim_contr_rep], dim=1).unsqueeze(0)
    tgt = tokenizer(rec_clauses, truncation=True, padding=True, return_tensors='pt', 
        return_token_type_ids=False, return_attention_mask=False).input_ids.transpose(0, 1)
    tgt_input = tgt[:-1, :]
    tgt_mask = generate_square_subsequent_mask(tgt_input.shape[0])
    tgt_padding_mask = (tgt_input == tokenizer.pad_token_id).transpose(0, 1)
    
    return ids, src, tgt, tgt_mask, tgt_padding_mask


def collate_only_contract_and_full_similar_contracts(batch, embed_type, contract_embeds, index, hnsw_k, tokenizer):
    ids = list()
    embeds_batch = list()
    rec_clauses = list()
    contr_names = list()

    for idx, embedding, _, rec_clause, contr in batch:
        ids.append(int(idx))
        embeds_batch.append(embedding)
        rec_clauses.append(clean_clause(rec_clause))
        contr_names.append(contr)
        
    query_contract_embeds = torch.cat(embeds_batch)
    sim_contr_rep = get_similar_contract_reps(index, hnsw_k, contract_embeds, query_contract_embeds, contr_names, embed_type)
    
    ids = torch.tensor(ids, dtype=torch.long)
    input_contr_ct_rep = torch.cat(embeds_batch)
    src = torch.cat([input_contr_ct_rep, sim_contr_rep], dim=1).unsqueeze(0)
    tgt = tokenizer(rec_clauses, truncation=True, padding=True, return_tensors='pt', 
        return_token_type_ids=False, return_attention_mask=False).input_ids.transpose(0, 1)
    tgt_input = tgt[:-1, :]
    tgt_mask = generate_square_subsequent_mask(tgt_input.shape[0])
    tgt_padding_mask = (tgt_input == tokenizer.pad_token_id).transpose(0, 1)
    
    return ids, src, tgt, tgt_mask, tgt_padding_mask
