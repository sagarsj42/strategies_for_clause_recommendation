import torch
from torch import nn, Tensor
from torch.nn import TransformerDecoderLayer, TransformerDecoder

from transformer_utils import TokenEmbedding, PositionalEncoding


class ClauseDecoder(nn.Module):
    def __init__(self, num_layers: int, emb_size: int, nhead: int, vocab_size: int, 
                 dim_feedforward: int=3072, dropout: float=0.1, activation: str='relu', batch_first: bool=True):
        
        super(ClauseDecoder, self).__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=nhead, 
                                                dim_feedforward=dim_feedforward, dropout=dropout, 
                                                activation=activation, batch_first=batch_first)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.generator = nn.Linear(in_features=emb_size, out_features=vocab_size, bias=True)
        
        self.xavier_initialize_for_decoder()
        
    
    def forward(self, src: Tensor, tgt: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor):
        tgt_emb = self.token_embedding(tgt)
        tgt_emb_pos = self.positional_encoding(tgt_emb)
        
        tgt_emb_out = self.decoder(tgt=tgt_emb_pos, memory=src, tgt_mask=tgt_mask, memory_mask=None, 
                               tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=None)
        
        output = self.generator(tgt_emb_out)
        
        return output
    
    
    def xavier_initialize_for_decoder(self):
        for p in self.decoder.named_parameters():
            if p[1].dim() > 1:
                nn.init.xavier_uniform_(p[1])
                
        return
    
    
    def print_params_info(self):
        print('%5s %55s %30s' % ('Parameter name', 'Shape', 'Requires grad'))
        for i, p in enumerate(self.named_parameters()):
            print(f'{i:5d} {p[0]:60s} {str(p[1].shape):30s} {str(p[1].requires_grad):30s}')
            
        return


class ClauseDecoderWithLinear(nn.Module):
    def __init__(self, num_layers: int, input_emb_size: int, emb_size: int, nhead: int, vocab_size: int, 
                 dim_feedforward: int=3072, dropout: float=0.1, activation: str='relu', batch_first: bool=True):
        
        super(ClauseDecoderWithLinear, self).__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        
        self.input_linear = nn.Linear(input_emb_size, emb_size)
        self.input_activation = nn.LeakyReLU(negative_slope=0.1)

        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=nhead, 
                                                dim_feedforward=dim_feedforward, dropout=dropout, 
                                                activation=activation, batch_first=batch_first)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.generator = nn.Linear(in_features=emb_size, out_features=vocab_size, bias=True)
        
        self.xavier_initialize_for_decoder()
        
    
    def forward(self, src: Tensor, tgt: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor):
        src = self.input_linear(src)
        src = self.input_activation(src)

        tgt_emb = self.token_embedding(tgt)
        tgt_emb_pos = self.positional_encoding(tgt_emb)
        
        tgt_emb_out = self.decoder(tgt=tgt_emb_pos, memory=src, tgt_mask=tgt_mask, memory_mask=None, 
                               tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=None)
        
        output = self.generator(tgt_emb_out)
        
        return output
    
    
    def xavier_initialize_for_decoder(self):
        for p in self.decoder.named_parameters():
            if p[1].dim() > 1:
                nn.init.xavier_uniform_(p[1])
                
        return
    
    
    def print_params_info(self):
        print('%5s %55s %30s' % ('Parameter name', 'Shape', 'Requires grad'))
        for i, p in enumerate(self.named_parameters()):
            print(f'{i:5d} {p[0]:60s} {str(p[1].shape):30s} {str(p[1].requires_grad):30s}')
            
        return


if __name__ == '__main__': 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE:', DEVICE)
    
    model = ClauseDecoder(3, 768, 8, 50000, 3072, 0.2, 'relu', False).to(DEVICE)
    model.print_params_info()

    model2 = ClauseDecoderWithLinear(3, 1536, 768, 8, 50000, 3072, 0.2, 'relu', False).to(DEVICE)
    model2.print_params_info()
