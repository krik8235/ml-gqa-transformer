import torch as t
import torch.nn as nn

from src.attention_layers import StandardAttention, MHA, MQA, GQA


class DecoderLayer(nn.Module):
    def __init__(self, attention_module: StandardAttention | MHA | MQA | GQA, d_model: int):
        super().__init__()
        # self attn block
        self.self_attn = attention_module # decoder self-attn (same as encoder self-attention)
        self.norm_self_attn = nn.LayerNorm(d_model)
        self.dropout_self_attn = nn.Dropout(0.1)

        # cross attn block
        self.cross_attn = MHA(d_model) # encoder-decoder attn
        self.norm_cross_attn = nn.LayerNorm(d_model)
        self.dropout_cross_attn = nn.Dropout(0.1)

        # ffn block
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))      
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout_ffn = nn.Dropout(0.1)


    def forward(self, tgt: t.Tensor, mem: t.Tensor) -> t.Tensor:  
        # self attn (q = k = v = tgt)
        norm_tgt_self_attn = self.norm_self_attn(tgt)
        self_attn_out = self.self_attn(norm_tgt_self_attn)
        tgt = tgt + self.dropout_self_attn(self_attn_out)

        # encoder-decoder cross-attn (q = tgt, k = v = mem)
        norm_tgt_cross_attn = self.norm_cross_attn(tgt)
        cross_attn_out = self.cross_attn(norm_tgt_cross_attn) 
        tgt = tgt + self.dropout_cross_attn(cross_attn_out)

        # ffn
        norm_tgt = self.norm_ffn(tgt)
        ffn_out = self.ffn(norm_tgt)
        output = tgt + self.dropout_ffn(ffn_out)
        return output
