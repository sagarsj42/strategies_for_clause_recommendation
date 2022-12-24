import math

import torch
from torch import nn, Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
        
    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float=0.1, max_len: int=5000):
        super(PositionalEncoding, self).__init__()
        
        div_term = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).unsqueeze(1)
        pos_embedding = torch.zeros((max_len, 1, emb_size))
        pos_embedding[:, 0, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 0, 1::2] = torch.cos(pos * div_term)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        
    def forward(self, token_embedding: Tensor) -> Tensor:
        x = token_embedding + self.pos_embedding[:token_embedding.size(0),:]
        
        return self.dropout(x)


if __name__ == '__main__':    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('DEVICE:', DEVICE)
    
    embedding = TokenEmbedding(5000, 200)
    print(embedding)
    
    positional_encoding = PositionalEncoding(200, 0.1, 3000)
    print(positional_encoding)
