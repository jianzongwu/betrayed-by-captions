import torch
from torch import nn

class BertEmbeddings(nn.Module):
    """Load word_embeddings and LayerNorm from huggingface BERT checkpoint to
    decrease the size of saved checkpoint
    """
    def __init__(self, bert_model):
        super().__init__()
        config = bert_model.config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.word_embeddings.load_state_dict(bert_model.embeddings.word_embeddings.state_dict())
        self.LayerNorm.load_state_dict(bert_model.embeddings.LayerNorm.state_dict())
        