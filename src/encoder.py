# encoder module + positional encoding
import math
import torch as t
import torch.nn as nn

from src.attention_layers import StandardAttention, MHA, MQA, GQA


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        # compute the positional encodings (pe) once in log space
        pe = t.zeros(max_len, d_model)
        position = t.arange(0, max_len, dtype=t.float).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, 2).float() * ( - math.log(10000.0) / d_model))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape - (1, max_len, d_model)

        self.register_buffer('pe', pe)
  
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Args:
            x: torch.Tensor, shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :] # type: ignore
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention_module: StandardAttention | MHA | MQA | GQA, d_model: int):
        super().__init__()

        # attn block
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = attention_module
        self.dropout_attn = nn.Dropout(0.1)

        # ffn block
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout_ffn = nn.Dropout(0.1)
    
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        # attention block
        norm_x = self.norm_attn(x)
        O = self.attn(norm_x)
        x = x + self.dropout_attn(O)

        # ffn block
        norm_x = self.norm_ffn(x)
        ffn_out = self.ffn(norm_x)
        output = x + self.dropout_ffn(ffn_out)
        return output