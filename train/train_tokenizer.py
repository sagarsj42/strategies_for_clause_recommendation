import os
import pickle

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders
from transformers import PreTrainedTokenizerFast


# Point to the path of a seriealized dataset
DATA_PATH = './clauserec-lbbase'

TOKENIZER_PATH = './clauserec-tokenizer-wordpiece.json'

UNK, UNK_IDX = '[UNK]', 0
BOS, BOS_IDX = '[BOS]', 1
EOS, EOS_IDX = '[EOS]', 2
PAD, PAD_IDX = '[PAD]', 3
MSK, MSK_IDX = '[MSK]', 4

train_files = os.listdir(os.path.join(DATA_PATH, 'train'))
train_files.sort(key=lambda f : int(f[:-4]))
train_op_clauses = list()

for train_file in train_files:
    with open(os.path.join(DATA_PATH, 'train', train_file), 'rb') as f:
        samples = pickle.load(f)
    for sample in samples:
        train_op_clauses.extend([c for _, c in sample[2]])

op_clause_string = ' '.join(train_op_clauses)
op_save_file = 'train-output-clauses.txt'
with open(op_save_file, 'w') as f:
    f.write(op_clause_string)

tokenizer = Tokenizer(WordPiece(unk_token=UNK))
tokenizer.pre_tokenizer = BertPreTokenizer()
tokenizer.post_processor = TemplateProcessing(single='[BOS] $0 [EOS]', special_tokens=[(BOS, BOS_IDX), (EOS, EOS_IDX)])
tokenizer.decoder = decoders.WordPiece()
tokenizer.enable_padding(pad_id=PAD_IDX, pad_token=PAD, pad_type_id=0)
tokenizer.enable_truncation(max_length=512)

trainer = WordPieceTrainer(vocab_size=8192, min_frequency=5, special_tokens=[UNK, BOS, EOS, PAD, MSK], show_progress=True)
tokenizer.train([op_save_file], trainer)
tokenizer.save(TOKENIZER_PATH)

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH, model_max_length = 512, 
    bos_token=BOS, eos_token=EOS, unk_token=UNK, sep_token=EOS, pad_token=PAD, cls_token=BOS, mask_token=MSK)
vocab = fast_tokenizer.get_vocab()

print('Vocab size:', len(vocab))
