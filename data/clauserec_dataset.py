import os
import pickle
from collections import defaultdict

from torch.utils.data import Dataset


class ClauserecDataset(Dataset):
    def __init__(self, dir_name, ids, split, embed_type='avg_sent_mean'):
        super(ClauserecDataset, self).__init__()
        self.path = os.path.join(dir_name, split)
        self.ids = list(ids[split])
        self.size = len(self.ids)
        self.embed_type = embed_type
        
        
    def __len__(self):
        return self.size
    
    
    def __getitem__(self, idx):            
        filename = self.ids[idx] + '.pkl'
        with open(os.path.join(self.path, filename), 'rb') as f:
            sample = pickle.load(f)
            sample_id = sample['id']
            embeds = sample[self.embed_type].cpu()
            label = sample['label']
            rec_clause = sample['rec_clause']
            contract = sample['contract']
            
        return (sample_id, embeds, label, rec_clause, contract)


def filter_clauserec_exclude_all_but_one(dir_name):
    ids = defaultdict(set)
    exclude_ids = set()
    
    for split in ('train', 'dev'):
        split_path = os.path.join(dir_name, split)
        split_files = os.listdir(split_path)
        
        for filename in split_files:
            sample_id = filename[:-4]
            
            if sample_id not in exclude_ids:
                ids[split].add(sample_id)
                with open(os.path.join(split_path, filename), 'rb') as f:
                    sample = pickle.load(f)
                    exclude_ids.update(list(map(str, sample['other_output_ids'])))
        ids[split] = list(ids[split])
                    
    return ids, exclude_ids
