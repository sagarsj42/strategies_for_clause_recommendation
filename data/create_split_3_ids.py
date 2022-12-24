import json

from sklearn.model_selection import StratifiedShuffleSplit

from clauserec_dataset import ClauserecDataset, filter_clauserec_exclude_all_but_one


DATA_PATH = 'clauserec-lbcontracts'

with open('label-clauses.json', 'r') as f:
    label_clauses = json.load(f)
LABELS = [label for label in label_clauses if len(label_clauses[label]) >= 5000]
LABELS.sort()

ids, _ = filter_clauserec_exclude_all_but_one(DATA_PATH)

print('ids len:', len(ids['train']), len(ids['dev']))

train_dataset = ClauserecDataset(DATA_PATH, ids, 'train')
dev_dataset = ClauserecDataset(DATA_PATH, ids, 'dev')

print('train/dev dataset sizes:', len(train_dataset), len(dev_dataset))

id_labels = [(str(dev_dataset[i][0]), dev_dataset[i][2]) for i in range(len(dev_dataset))]

print('len of id_labels in dev:', len(id_labels))

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
labs = [l for _, l in id_labels]

for dev_index, test_index in sss.split(id_labels, labs):
    dev_ids = [id_labels[idx][0] for idx in dev_index]
    test_ids = [id_labels[idx][0] for idx in test_index]

dev_ids = set(dev_ids)
test_ids = set(test_ids)

print('dev/test id sizes:', len(dev_ids), len(test_ids))

dev_lcounts = dict()
test_lcounts = dict()

for idx, label in id_labels:
    if idx in dev_ids:
        if label in dev_lcounts:
            dev_lcounts[label] += 1
        else:
            dev_lcounts[label] = 1
    elif idx in test_ids:
        if label in test_lcounts:
            test_lcounts[label] += 1
        else:
            test_lcounts[label] = 1

print('dev label counts:', dev_lcounts, 'test label counts:', test_lcounts)

for label in LABELS:
    print(label, dev_lcounts[label], test_lcounts[label])

split3_ids = {
    'train': ids['train'],
    'dev': list(dev_ids),
    'test': list(test_ids)
}

print('len of split3 ids:', len(split3_ids['train']), len(split3_ids['dev']), len(split3_ids['test']))

with open(DATA_PATH + '-3split-ids.json', 'w') as f:
    json.dump(split3_ids, f)
