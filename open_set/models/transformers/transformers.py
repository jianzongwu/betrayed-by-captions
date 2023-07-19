import numpy as np 
import operator as op 
import itertools as it, functools as ft

import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, in_dim, drop_val=0.1):
        super(PositionalEncoding, self).__init__()
        pos = np.arange(0, seq_length)[:, None]
        idx = np.fromfunction(lambda _,j: j - j % 2, shape=(1, in_dim))
        mask = np.fromfunction(lambda _,j: j % 2 == 0, shape=(1, in_dim))

        pnt = pos / (10000 ** (idx / in_dim))
        val = np.sin(pnt) * mask + np.cos(pnt) * (1 - mask)

        self.drop_layer = nn.Dropout(drop_val)
        self.register_buffer('psne_layer', th.tensor(val).float())

    def forward(self, src):
        _, seq_length, _ = src.shape
        pos = self.psne_layer[:seq_length, :][None, ...]
        return self.drop_layer(src + pos)

class FeedForwardNetwork(nn.Module):
    __THETA = {  # map id to non_linear
        0: nn.Identity(), 
        1: nn.ReLU(),
        2: nn.GELU(),
        3: nn.Sigmoid(),
        4: nn.Tanh(),
        5: nn.Softmax(dim=-1)
    }
    def __init__(self, layer_cfg, activations, drop_vals):
        super(FeedForwardNetwork, self).__init__()
        self.shapes = list(zip(layer_cfg[:-1], layer_cfg[1:]))
        self.linears = nn.ModuleList([])
        for idx, (in_dim, out_dim) in enumerate(self.shapes):
            fn_id = activations[idx]
            proba = drop_vals[idx]
            block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(proba) if proba > 0.0 else nn.Identity(),
                FeedForwardNetwork.__THETA.get(fn_id, nn.Identity())
            )
            self.linears.append(block)
    
    def forward(self, input_batch):
        output_batch = ft.reduce(  # functools 
            lambda acc, crr: crr(acc),
            self.linears, 
            input_batch
        )
        return output_batch
        
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, in_dim, nb_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.nbr_heads = nb_heads 
        self.heads_dim = in_dim // nb_heads

        self.to_qry = nn.Linear(in_dim, in_dim)
        self.to_key = nn.Linear(in_dim, in_dim)
        self.to_val = nn.Linear(in_dim, in_dim)
        self.to_out = nn.Linear(in_dim, in_dim)
    
    def __rearrange(self, seq):
        bt_size, seq_length, _ = seq.shape  # unpack shape 
        seq = seq.reshape(bt_size, seq_length, self.nbr_heads, self.heads_dim).permute(0, 2, 1, 3)
        return seq 

    def forward(self, qry, key, val, mask=None, key_padding_mask=None):
        
        qry = self.to_qry(qry)
        key = self.to_key(key)
        val = self.to_val(val)

        qry = self.__rearrange(qry)
        key = self.__rearrange(key)
        val = self.__rearrange(val)

        dim = qry.shape[-1]

        wgt = qry @ key.transpose(-2, -1)
        wgt = wgt / np.sqrt(dim)
        if mask is not None:
            wgt = wgt.masked_fill(mask, float('-inf'))
        if key_padding_mask is not None:
            cnd = key_padding_mask[:, None, None, :]
            wgt = wgt.masked_fill(cnd, float('-inf'))
        wgt = th.softmax(wgt, dim=-1)
        
        res = wgt @ val
        res = res.permute(0, 2, 1, 3)  # permute head and sequence 
        res = th.flatten(res, start_dim=2)  # concat over heads 
        res = self.to_out(res)
        
        return res 

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, nb_heads):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.nbr_heads = nb_heads 
        self.heads_dim = in_dim // nb_heads 
        self.qkv_layer = nn.Linear(in_dim, 3 * in_dim)
        self.out_layer = nn.Linear(in_dim, in_dim)
    
    def forward(self, src, mask=None, key_padding_mask=None):
        bt_size, seq_length, _ = src.shape  # unpack shape 
        
        qkv = self.qkv_layer(src)  # extract query, key and value  
        qkv = qkv.reshape(bt_size, seq_length, self.nbr_heads, 3 * self.heads_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # permute head and sequence
        qry, key, val = th.chunk(qkv, 3, dim=-1)

        dim = qry.shape[-1]
        wgt = qry @ key.transpose(-2, -1)  # hidden_dim and sequence_dim
        wgt = wgt / np.sqrt(dim)  # normalize 
        if mask is not None:
            wgt = wgt.masked_fill(mask, float('-inf'))
        if key_padding_mask is not None:
            cnd = key_padding_mask[:, None, None, :]
            wgt = wgt.masked_fill(cnd, float('-inf'))
        wgt = th.softmax(wgt, dim=-1)

        res = wgt @ val 
        res = res.permute(0, 2, 1, 3)  # permute head and sequence 
        res = th.flatten(res, start_dim=2)  # concat over heads 
        res = self.out_layer(res)

        return res 

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, ff_dim, nb_heads, drop_val=0.1, pre_norm=False):
        super(EncoderBlock, self).__init__()
        assert in_dim % nb_heads == 0 
        
        self.nbr_heads = nb_heads 
        self.heads_dim = in_dim // nb_heads 

        self.mha_layer = MultiHeadSelfAttention(in_dim, nb_heads)
        self.ffn_layer = FeedForwardNetwork([in_dim, ff_dim, in_dim], [1, 0], [drop_val, 0.0])

        self.dropout_layer = nn.ModuleDict({
            'mha': nn.Dropout(drop_val),
            'ffn': nn.Dropout(drop_val)
        })
        self.layer_normalz = nn.ModuleDict({
            'mha': nn.ModuleList([
                nn.LayerNorm(in_dim) if pre_norm else nn.Identity(),
                nn.LayerNorm(in_dim) if not pre_norm else nn.Identity()
            ]), 
            'ffn': nn.ModuleList([
                nn.LayerNorm(in_dim) if pre_norm else nn.Identity(),
                nn.LayerNorm(in_dim) if not pre_norm else nn.Identity()
            ])
        })
        

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # multi head self attention
        tmp = self.layer_normalz['mha'][0](src) 
        out = self.mha_layer(tmp, src_mask, src_key_padding_mask)
        out = self.dropout_layer['mha'](out)
        agg = tmp + out 
        agg = self.layer_normalz['mha'][1](agg) 

        # feed forward network 
        tmp = self.layer_normalz['ffn'][0](agg)
        out = self.ffn_layer(tmp)
        out = self.dropout_layer['ffn'](out)
        agg = tmp + out 
        agg = self.layer_normalz['ffn'][1](agg)

        return agg   

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, ff_dim, nb_heads, drop_val=0.1, pre_norm=False):
        super(DecoderBlock, self).__init__()
        assert in_dim % nb_heads == 0 

        self.nbr_heads = nb_heads 
        self.heads_dim = in_dim // nb_heads 

        self.mha_layer = MultiHeadSelfAttention(in_dim, nb_heads)
        self.crx_layer = MultiHeadCrossAttention(in_dim, nb_heads)
        self.ffn_layer = FeedForwardNetwork([in_dim, ff_dim, in_dim], [1, 0], [drop_val, 0.0])

        self.dropout_layer = nn.ModuleDict({
            'mha': nn.Dropout(drop_val),
            'crx': nn.Dropout(drop_val),
            'ffn': nn.Dropout(drop_val)
        })
        self.layer_normalz = nn.ModuleDict({
            'mha': nn.ModuleList([
                nn.LayerNorm(in_dim) if pre_norm else nn.Identity(),
                nn.LayerNorm(in_dim) if not pre_norm else nn.Identity() 
            ]), 
            'crx': nn.ModuleList([
                nn.LayerNorm(in_dim) if pre_norm else nn.Identity(),
                nn.LayerNorm(in_dim) if not pre_norm else nn.Identity()
            ]),
            'ffn': nn.ModuleList([
                nn.LayerNorm(in_dim) if pre_norm else nn.Identity(),
                nn.LayerNorm(in_dim) if not pre_norm else nn.Identity()
            ])
        })

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # masked multi head attention 
        tmp = self.layer_normalz['mha'][0](tgt)
        out = self.mha_layer(tmp, tgt_mask, tgt_key_padding_mask)
        out = self.dropout_layer['mha'](out)
        agg = tmp + out  # residual 
        agg = self.layer_normalz['mha'][1](agg)

        # cross multi head attention 
        tmp = self.layer_normalz['crx'][0](agg)
        out = self.crx_layer(tmp, memory, memory, memory_mask, memory_key_padding_mask)
        out = self.dropout_layer['crx'](out)
        agg = tmp + out  # residual 
        agg = self.layer_normalz['crx'][1](agg)

        # feed forward network 
        tmp = self.layer_normalz['ffn'][0](agg)
        out = self.ffn_layer(agg)
        out = self.dropout_layer['ffn'](out)
        agg = tmp + out  # residual 
        agg = self.layer_normalz['ffn'][1](agg)

        return agg  

class TransformerEncoder(nn.Module):
    def __init__(self, nb_layers, in_dim, ff_dim, nb_heads, drop_val=0.1, pre_norm=False):
        super(TransformerEncoder, self).__init__()
        self.encoders = nn.ModuleList([])
        for _ in range(nb_layers):
            blk = EncoderBlock(in_dim=in_dim, ff_dim=ff_dim, nb_heads=nb_heads, drop_val=drop_val, pre_norm=pre_norm)
            self.encoders.append(blk)
    
    def forward(self, src, mask=None, key_padding_mask=None):
        fnl = ft.reduce(
            lambda acc, crr: acc + [crr(acc[-1], mask, key_padding_mask)], 
            self.encoders, 
            [src]
        )
        return fnl[1:]  # ignore src 

class TransformerDecoder(nn.Module):
    def __init__(self, nb_layers, in_dim, ff_dim, nb_heads, drop_val=0.1, pre_norm=False):
        super(TransformerDecoder, self).__init__()
        self.decoders = nn.ModuleList([])
        for _ in range(nb_layers):
            blk = DecoderBlock(in_dim=in_dim, ff_dim=ff_dim, nb_heads=nb_heads, drop_val=drop_val, pre_norm=pre_norm)
            self.decoders.append(blk)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        lng = len(memory) - 1 
        fnl = ft.reduce(
            lambda acc,crr: acc + [crr[1](acc[-1], memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)],
            enumerate(self.decoders),
            [tgt]
        )
        return fnl[1:]  # ignore tgt

class Transformer(nn.Module):    
    def __init__(self, in_dim, ff_dim, nb_heads, encoder_depth, decoder_depth, drop_val=0.1, pre_norm=False):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(
            nb_layers=encoder_depth, 
            in_dim=in_dim,
            ff_dim=ff_dim,
            nb_heads=nb_heads,
            drop_val=drop_val,
            pre_norm=pre_norm 
        )
        self.decoder = TransformerDecoder(
            nb_layers=decoder_depth, 
            in_dim=in_dim,
            ff_dim=ff_dim,
            nb_heads=nb_heads,
            drop_val=drop_val,
            pre_norm=pre_norm
        )
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return output 