import numpy as np
import torch
import torch.nn as nn 

from .transformers import TransformerDecoder, PositionalEncoding
from mmdet.models.builder import HEADS

def build_mask(seq):
    seq_length = seq.shape[1]
    mask = np.fromfunction(lambda i,j: j > i, shape=(seq_length, seq_length))
    return torch.as_tensor(mask) 

def build_key_padding_mask(seq, pad_idx):
    seq_key_padding_mask = (seq == pad_idx)
    return seq_key_padding_mask

@HEADS.register_module()
class CaptionTransformer(nn.Module):
    def __init__(self, nb_layers, input_dim, hidden_dim, ff_dim, nb_heads, drop_val, pre_norm, seq_length, nb_tokens):
        super(CaptionTransformer, self).__init__()
        if input_dim != hidden_dim:
            self.adapter = nn.Linear(input_dim, hidden_dim)
        else:
            self.adapter = nn.Identity()
        self.position_encoder = PositionalEncoding(seq_length, hidden_dim)
        self.transformer_decoder = TransformerDecoder(
            nb_layers=nb_layers,
            in_dim=hidden_dim,
            ff_dim=ff_dim,
            nb_heads=nb_heads,
            drop_val=drop_val,
            pre_norm=pre_norm
        )
        self.generator = nn.Linear(hidden_dim, nb_tokens)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.adapter(memory)
        tgt = self.position_encoder(tgt)
        if tgt_mask is None:
            tgt_mask = build_mask(tgt).to(tgt.device)

        output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        last_layer_logits = self.generator(output[-1])
        return output, last_layer_logits