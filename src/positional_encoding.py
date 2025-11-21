import math
import torch as t
import torch.nn as nn


# fixed absolute pe
class FAPE(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        pe = t.zeros(max_seq_len, d_model)
        position = t.arange(0, max_seq_len, dtype=t.float).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, 2).float() * ( - math.log(10000.0) / d_model))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
  
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x + self.pe[:, :x.size(1), :] # type: ignore
        return x


# learnable pe
class LPE(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.pe = nn.Parameter(t.randn(max_seq_len, d_model))
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x + self.pe[:x.size(1), :].unsqueeze(0)