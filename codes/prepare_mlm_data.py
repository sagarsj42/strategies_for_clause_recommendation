import os
os.chdir('working_dir')

import pandas as pd
import nltk
from ml_things import fix_text

filename = 'LEDGAR_2016-2019_clean.jsonl'
ledgar = pd.read_json(filename, lines=True)

clauses = ledgar['provision'].tolist()

clause_sents = list()

for clause in clauses:
    sents = nltk.sent_tokenize(clause)
    for sent in sents:
        sent = fix_text(sent).lower()
        clause_sents.append(sent)

divider = int(len(clause_sents) * 0.8)
train_sents = clause_sents[:divider]
valid_sents = clause_sents[divider:]

with open('ledgar-clause-sent-train.txt', 'w') as f:
    for s in train_sents:
        f.write(s + '\n')
    
with open('ledgar-clause-sent-valid.txt', 'w') as f:
    for s in valid_sents:
        f.write(s + '\n')
