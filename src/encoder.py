# encoder module
import torch as t
import torch.nn as nn

from src.attention_layers import StandardAttention, MHA, MQA, GQA



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