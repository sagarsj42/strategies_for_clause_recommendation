import os

# You may change to a custom working directory for storing your data and any transformer files here
os.chdir('/working_dir')
os.environ['TORCH_HOME'] = 'working_dir'
os.environ['TRANSFORMERS_CACHE'] = 'working_dir'

import time
import math
import json
import pickle
from functools import partial

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

import wandb
from accelerate import Accelerator

from clause_decoder import ClauseDecoder, ClauseDecoderWithLinear
from clauserec_dataset import ClauserecDataset
from indexing import construct_index
from clauserec_collators import collate_only_contract_with_tokenizer, collate_with_label_and_tokenizer, \
    collate_with_label_and_full_similar_contracts, collate_with_label_and_serialized_select_clause_rep_from_similar_contracts, \
    collate_only_contract_and_full_similar_contracts
from clauserec_utils import get_wordpiece_tokenizer


def train_epoch(train_dataloader, model, optimizer, scheduler, loss_fn, accelerator, train_info, logger):
    model.train()
    curr_epoch = train_info['curr_epoch']
    step_loss = 0.0
    total_loss = 0.0
    n_steps = len(train_dataloader)
    start_time = time.time()
    optimizer.zero_grad()
    
    for i, (_, src, tgt, tgt_mask, tgt_padding_mask) in enumerate(train_dataloader):
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]
        
        with accelerator.autocast():
            logits = model(src=src, tgt=tgt_input, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 2.0, norm_type=2)

        if (i+1) % train_info['accumulate_train_batches'] == 0 or (i+1) == n_steps:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step_loss += loss.item()
        total_loss += loss.item()

        if accelerator.is_local_main_process:
            lr = optimizer.param_groups[0]['lr']
            if (i+1) % train_info['log_steps'] == 0:
                cur_loss = step_loss / train_info['log_steps']
                ppl = math.exp(cur_loss)
                ms_per_batch = (time.time() - start_time) * 1000 / train_info['log_steps']
                
                train_info['curr_step'] = i+1
                train_info['avg_step_train_losses'].append(cur_loss)
                train_info['avg_ms_per_batch'].append(ms_per_batch)

                accelerator.print(f'| epoch {curr_epoch:3d} | step {i+1:5d} / {n_steps:5d} batches '
                    f'| milli-sec/batch {ms_per_batch:7.2f} | loss {cur_loss:7.2f} | ppl {ppl:8.2f} |')

                logger.log({'lr': lr, 'train/step/#': (i+1), 'train/step/loss': cur_loss, 'train/step/ppl': ppl, 
                    'train/step/ms_per_batch': ms_per_batch})

                step_loss = 0.0
                start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    
    return total_loss / n_steps, lr


@torch.no_grad()
def evaluate(dev_dataloader, model, loss_fn, accelerator):
    model.eval()
    losses = 0.0
    
    for _, src, tgt, tgt_mask, tgt_padding_mask in dev_dataloader:
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]
        logits = model(src=src, tgt=tgt_input, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        loss_mean = accelerator.gather(loss).mean()
        losses += loss_mean.item()
        
    return losses / len(dev_dataloader)


def save_checkpoint(filename, model, optimizer, accelerator, save_dict, store_optimizer_state=False):
    os.makedirs(EXP_NAME, exist_ok=True)
    
    unwrapped_model = accelerator.unwrap_model(model)
    save_dict['model_state_dict'] = unwrapped_model.state_dict()
    save_dict['optimizer_args'] = OPTIMIZER_ARGS
    if store_optimizer_state:
        unwrapped_optimizer = accelerator.unwrap_model(optimizer)
        save_dict['optimizer_state_dict'] = unwrapped_optimizer.state_dict()
    accelerator.save(save_dict, os.path.join(EXP_NAME, filename))
    
    return


def train(train_dataloader, dev_dataloader, model, loss_fn, optimizer, scheduler, accelerator, save_dict, logger):
    train_info = save_dict['train_logs']
    for epoch in range(1, train_info['total_epochs']+1):
        epoch_start_time = time.time()
        train_info['curr_epoch'] = epoch
        
        train_start_time = time.time()
        train_loss, lr = train_epoch(train_dataloader, model, optimizer, scheduler, loss_fn, accelerator, train_info, logger)
        train_duration = time.time() - train_start_time

        accelerator.wait_for_everyone()
        accelerator.print(f'Training complete for epoch {epoch} with average loss: {train_loss}, current learning rate {lr}')
        accelerator.print('Starting validation')
        
        dev_start = time.time()
        dev_loss = evaluate(dev_dataloader, model, loss_fn, accelerator)
        dev_ppl = math.exp(dev_loss)
        
        if accelerator.is_local_main_process:
            if dev_loss < train_info['best_eval_loss']:
                train_info['best_eval_loss'] = dev_loss
                save_checkpoint('best.pth', model, optimizer, accelerator, save_dict, store_optimizer_state=False)
                accelerator.print('*'*10, 'Updated as best checkpoint', '*'*10)
                
            eval_duration = time.time() - dev_start
            train_info['avg_eval_losses'].append(dev_loss)
            train_info['eval_durations'].append(eval_duration)
            
            accelerator.print(f'|| epoch {epoch:3d} | '
                    f'| eval duration {eval_duration:7.2f} sec | eval loss {dev_loss:5.2f} '
                    f'| eval ppl {dev_ppl:8.2f} ||')

            logger.log({'train/epoch/duration': train_duration, 'train/epoch/loss': train_loss, 'dev/epoch/duration': eval_duration, 
                'dev/epoch/loss': dev_loss, 'dev/epoch/ppl': dev_ppl})
            
            epoch_duration = time.time() - epoch_start_time
            train_info['epoch_durations'].append(epoch_duration)
            save_checkpoint(f'epoch-{epoch}.pth', model, optimizer, accelerator, save_dict, store_optimizer_state=False)
        
            accelerator.print('-'*90)
            accelerator.print(f'| end of epoch {epoch:3d} | time: {epoch_duration:7.2f}s | train loss {train_loss:5.2f} '
                f'| eval loss {dev_loss:5.2f} | eval ppl {dev_ppl:8.2f} |')
            accelerator.print('-'*90)
            logger.log({'epoch/#': epoch, 'epoch/duration': epoch_duration})

    return


if __name__ == '__main__':
    accelerator = Accelerator()
    torch.manual_seed(0)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accelerator.print('device:', DEVICE)

    # Custom name of the experiment.
    # There will be a folder created with this name, in which the checkpoints will be saved.
    EXP_NAME = 'clausedecoder-lbbase-avg_clause_mean-only_contract'

    # Path to the serialized dataset w.r.t. the base encoder chosen
    DATA_PATH = './clauserec-lbbase'
    
    # Path to the serialzied clause type embeddings w.r.t. the base encoder
    LABEL_EMBEDS = 'label-embeddings-lbbase.pkl'
    
    # Path to the serialized contract embeddings w.r.t. the base encoder
    CONTRACT_EMBEDS = 'contract-embeddings-lbbase.pkl'
    
    # The strategy to average the representation
    # Can take 4 values: avg_sent_mean, avg_sent_cls, avg_clause_mean, avg_clause_cls
    # The function create_clauserec_embeddings_dataset in create_clauserec_embeddings_dataset.py introduces the 4 ways
    EMBED_TYPE = 'avg_clause_mean'
    
    # The strategy for selecting how do we create the context for representation.
    # The strategies available and their corresponding names in the paper:
    # only_contract: ONLY_CONTR
    # contract_and_label: CONTR_TYPE
    # contract_label_fullsim_contract: CONTR_FULLSIM
    # contract_label_sim_contr_clauses: CONTR_TYPE_FULLSIM
    # only_contract_fullsim_contract: CONTR_TYPE_CLAUSESIM
    STRATEGY = 'only_contract'
    
    # The # retrieved similar contracts. Set it to K+1
    RETRIEVE_K = 6

    # Use ClauseDecoderWithLinear for strategies involving similar contracts, else ClauseDecoder
    MODEL_NAME = 'ClauseDecoder'
    DECODER_VOCAB_SIZE = -1
    EMB_SIZE = 768
    
    # Uncomment this line for the strategy involving similar contracts
    # INPUT_EMB_SIZE = 2 * EMB_SIZE
    
    DECODER_NLAYERS = 3
    DECODER_NHEADS = 12
    DECODER_FFN_HID_DIM = 3072

    TRAIN_BATCH_SIZE = 19
    DEV_BATCH_SIZE = 64
    ACCUMULATE_TRAIN_BATCHES = 4
    N_EPOCHS = 50

    if 'mlm' in DATA_PATH:
        SIM_CLAUSE_EMB_FILE = 'simclause-embeddings-bertmlm.pkl'
    elif 'lbbase' in DATA_PATH:
        SIM_CLAUSE_EMB_FILE = 'simclause-embeddings-lbbase.pkl'
    elif 'lbcontracts' in DATA_PATH:
        SIM_CLAUSE_EMB_FILE = 'simclause-embeddings-lbcontracts.pkl'

    DATA_INFO = {
        'dataset': os.path.basename(DATA_PATH),
        'embed_type': EMBED_TYPE,
        'strategy': STRATEGY,
        'train_size': 0,
        'dev_size': 0
    }

    INDEXING_INFO = {
        'indexing_strategy': 'faiss.IndexHNSWFlat',
        'd': EMB_SIZE,
        'm': 64,
        'ef_construction': 128,
        'ef_search': 64,
        'retrieve_k': RETRIEVE_K
    }

    MODEL_ARGS = {
        'num_layers': DECODER_NLAYERS,
        
        # Uncomment this line for the strategy involving similar contracts
        # 'input_emb_size': INPUT_EMB_SIZE,
        
        'emb_size': EMB_SIZE,
        'nhead': DECODER_NHEADS,
        'vocab_size': DECODER_VOCAB_SIZE,
        'dim_feedforward': DECODER_FFN_HID_DIM,
        'dropout': 0.2,
        'activation': 'relu',
        'batch_first': False
    }

    OPTIMIZER_ARGS = {
        'lr': 6e-5,
        'betas': (0.9, 0.98),
        'weight_decay': 1e-2,
        'eps': 1e-9
    }

    TRAIN_LOGS = {
        'per_device_train_batch_size': TRAIN_BATCH_SIZE,
        'per_device_dev_batch_size': DEV_BATCH_SIZE,
        'accumulate_train_batches': ACCUMULATE_TRAIN_BATCHES,
        'total_epochs': N_EPOCHS,
        'log_steps': 50,
        'eval_steps': -1,
        'curr_epoch': 0,
        'curr_step': 0,
        'best_eval_loss': float('inf'),
        'avg_step_train_losses': list(),
        'avg_eval_losses': list(),
        'avg_ms_per_batch': list(),
        'eval_durations': list(),
        'epoch_durations': list()
    }

    COMMENTS = 'Experiment on full filtered dataset by randomly choosing one output in case multiple per contract, discarding the rest'

    SAVE_DICT = {
        'experiment_name': EXP_NAME,
        'model_name': MODEL_NAME,
        'model_args': MODEL_ARGS,
        'model_state_dict': {},
        'optimizer_args': OPTIMIZER_ARGS,
        'scheduler_args': {},
        'data_info': DATA_INFO,
        'indexing_info': INDEXING_INFO,
        'train_logs': TRAIN_LOGS,
        'comments': COMMENTS
    }

    tokenizer = get_wordpiece_tokenizer()
    MODEL_ARGS['vocab_size'] = len(tokenizer.get_vocab())
    
    # Use ClauseDecoderWithLinear for strategies involving similar contracts, else ClauseDecoder
    model = ClauseDecoder(**MODEL_ARGS)
    if accelerator.is_local_main_process:
        model.print_params_info()

    with open(DATA_PATH + '-3split-ids.json', 'r') as f:
        ids = json.load(f)

    train_dataset = ClauserecDataset(DATA_PATH, ids, 'train', EMBED_TYPE)
    dev_dataset = ClauserecDataset(DATA_PATH, ids, 'dev', EMBED_TYPE)

    # Uncomment these lines to run the code on a small subset for verification
    # train_dataset = Subset(train_dataset, list(range(1000)))
    # dev_dataset = Subset(dev_dataset, list(range(200)))

    with open(LABEL_EMBEDS, 'rb') as f:
        label_embeds = pickle.load(f)

    with open(CONTRACT_EMBEDS, 'rb') as f:
        contract_embeds = pickle.load(f)

    with open(SIM_CLAUSE_EMB_FILE, 'rb') as f:
        simclause_emb = pickle.load(f)

    index = construct_index(contract_embeds, EMBED_TYPE, INDEXING_INFO['d'], INDEXING_INFO['m'], 
        INDEXING_INFO['ef_construction'], INDEXING_INFO['ef_search'])

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

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=DEV_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    DATA_INFO['train_size'] = len(train_dataset)
    DATA_INFO['dev_size'] = len(dev_dataset)
    DATA_INFO['total_train_steps'] = len(train_dataloader)
    DATA_INFO['total_dev_steps'] = len(dev_dataloader)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), **OPTIMIZER_ARGS)

    train_dataloader, dev_dataloader = accelerator.prepare(train_dataloader, dev_dataloader)

    num_train_update_steps = N_EPOCHS * len(train_dataloader) // ACCUMULATE_TRAIN_BATCHES
    SCHEDULER_ARGS = {
        'num_warmup_steps': num_train_update_steps // 4,
        'num_training_steps': num_train_update_steps,
    }
    SAVE_DICT['scheduler_args'] = SCHEDULER_ARGS
    scheduler = get_linear_schedule_with_warmup(optimizer, **SCHEDULER_ARGS)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    DATA_INFO['per_device_train_steps'] = len(train_dataloader)
    DATA_INFO['per_device_dev_steps'] = len(dev_dataloader)

    accelerator.print('State:', SAVE_DICT)

    if accelerator.is_local_main_process:
        logger = wandb.init(project=EXP_NAME, config=SAVE_DICT)
        logger.watch(model)
        train(train_dataloader, dev_dataloader, model, loss_fn, optimizer, scheduler, accelerator, SAVE_DICT, logger)
    else:
        train(train_dataloader, dev_dataloader, model, loss_fn, optimizer, scheduler, accelerator, SAVE_DICT, None)

    wandb.finish()
