import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from utils.clauserec_utils import generate_square_subsequent_mask


@torch.no_grad()
def greedy_decode_with_tokenizer(model: nn.Module, memory: Tensor, tokenizer: PreTrainedTokenizerFast, device='cpu'):
    if len(memory.shape) == 2:
        memory = memory.unsqueeze(dim=0)
        
    memory = memory.to(device)
    ys = torch.ones(1, 1).fill_(tokenizer.bos_token_id).type(torch.long).to(device)
    
    for _ in range(tokenizer.model_max_length-1):
        if ys[-1] == tokenizer.eos_token_id:
            break
        
        ys = ys.to(device)
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool).to(device)
        tgt_padding_mask = (ys == tokenizer.pad_token_id).transpose(0, 1).to(device)
        
        prob = model(src=memory, tgt=ys, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask)
        next_token = torch.argmax(prob[-1, :, :]).item()
        ys = torch.cat([ys, torch.ones(1, 1).fill_(next_token).type(torch.long).to(device)], dim=0)
    token_output = ys.transpose(0, 1)

    return token_output


@torch.no_grad()
def greedy_decode_batch_with_tokenizer(model: nn.Module, memory: Tensor, tokenizer: PreTrainedTokenizerFast, device='cpu'):
    if len(memory.shape) == 2:
        memory = memory.unsqueeze(dim=0)
        
    memory = memory.to(device)
    curr_size = memory.shape[1]
    ys = torch.ones(1, curr_size).fill_(tokenizer.bos_token_id).type(torch.long).to(device)
    total_done = set()
    
    for _ in range(tokenizer.model_max_length-1):
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool).to(device)
        tgt_padding_mask = (ys == tokenizer.pad_token_id).transpose(0, 1).to(device)
        
        prob = model(src=memory, tgt=ys, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask).detach()
        next_token = torch.argmax(prob[-1, :, :], dim=1).unsqueeze(0)
        ys = torch.cat([ys, next_token]).to(device)
        find = (next_token == tokenizer.eos_token_id).nonzero()
        
        if find.size(0) > 0:
            total_done = total_done.union(set(find[:, 1].tolist()))
        if len(total_done) == curr_size:
            break
    token_output = ys.transpose(0, 1)
    
    return token_output


def get_model_output_with_tokenizer(model: nn.Module, embed: Tensor, tokenizer: PreTrainedTokenizerFast, 
    batch_size: int=1, replace_specials=True, device='cpu'):
    
    if batch_size == 1:
        token_output = greedy_decode_with_tokenizer(model, embed, tokenizer, device)
        output = tokenizer.decode(token_output, skip_special_tokens=replace_specials)
    else:
        token_output = greedy_decode_batch_with_tokenizer(model, embed, tokenizer, device)
        output = tokenizer.batch_decode(token_output, skip_special_tokens=replace_specials)

    return output


def get_model_outputs_with_tokenizer(dataloader: DataLoader, tokenizer: PreTrainedTokenizerFast, model, batch_size, device='cpu'):
    all_outputs = list()
    all_actuals = list()

    for i, (_, src, tgt, _, _) in enumerate(dataloader):
        print(src.shape, tgt.shape)
        outputs = get_model_output_with_tokenizer(model, src, tokenizer=tokenizer, batch_size=batch_size, replace_specials=True, device=device)
        actual_strings = tokenizer.batch_decode(tgt.transpose(0, 1), skip_special_tokens=True)
        
        all_outputs.extend(outputs)
        all_actuals.extend(actual_strings)
        
        if (i+1) % 10 == 0:
            print(i+1, 'done')
        
    return all_outputs, all_actuals
