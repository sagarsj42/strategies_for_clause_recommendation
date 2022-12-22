import faiss
import torch


def construct_index(contract_embeds, embed_type, hnsw_d, hnsw_m, hnsw_ef_construction, hnsw_ef_search):
    contract_embeds_array = torch.cat([ce[embed_type] for ce in contract_embeds], dim=0).numpy()
    
    index = faiss.IndexHNSWFlat(hnsw_d, hnsw_m, faiss.METRIC_L2)
    index.hnsw.efConstruction = hnsw_ef_construction
    index.hnsw.efSearch = hnsw_ef_search
    index.add(contract_embeds_array)

    return index


def get_similar_contract_reps(index, hnsw_k, full_contr_emb, query_contr_embs, contr_names, emb_type):
    query_contr_embs_array = query_contr_embs.numpy()
    _, output_i = index.search(query_contr_embs_array, hnsw_k)
    op_index_sets = output_i.tolist()
    sim_contr_emb = torch.zeros_like(query_contr_embs)
    
    for i, (contr_name, op_indices) in enumerate(zip(contr_names, op_index_sets)):
        net_embed = torch.zeros_like(query_contr_embs[0].unsqueeze(0))
        added = 0
        for o_i in op_indices:
            contr_retr = full_contr_emb[o_i]['contract']
            if contr_retr != contr_name:
                net_embed = torch.add(net_embed, full_contr_emb[o_i][emb_type])
                added += 1
        net_embed /= added
        sim_contr_emb[i, :] = net_embed

    return sim_contr_emb


def get_serialized_select_clause_rep_from_similar_contract_reps(index, hnsw_k, full_contr_emb, query_contr_embs, 
    contr_names, simclause_emb, batch_labels):
    
    query_contr_embs_array = query_contr_embs.numpy()
    _, output_i = index.search(query_contr_embs_array, hnsw_k)
    op_index_sets = output_i.tolist()
    sim_contr_emb = torch.zeros_like(query_contr_embs)
    
    for i, (contr_name, label, op_indices) in enumerate(zip(contr_names, batch_labels, op_index_sets)):
        net_embed = torch.zeros_like(query_contr_embs[0].unsqueeze(0))
        added = 0
        for o_i in op_indices:
            contr_retr = full_contr_emb[o_i]['contract']
            if contr_retr != contr_name:
                if label in simclause_emb[contr_retr]:
                    clause_emb = simclause_emb[contr_retr][label]
                    net_embed = torch.add(net_embed, clause_emb)
                    added += 1
        added = max(1, added)
        net_embed /= added
        sim_contr_emb[i, :] = net_embed

    return sim_contr_emb
