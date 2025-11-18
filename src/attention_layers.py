from typing import Optional
import torch as t
import torch.nn as nn
import torch.nn.functional as F


class StandardAttention(nn.Module):
    def __init__(self, d_model: int = 512, d_V: int = 64, projection: bool = True) -> None: # d_model - dim of the model input layer, d_v - dim of value vector
        super().__init__()
        self.d_model = d_model
        self.d_V = d_V
        self.projection = projection
        self.scaling_factor = 1 / t.sqrt(t.tensor(self.d_V, requires_grad=False))
        self.query = nn.Linear(in_features=self.d_model, out_features=self.d_V, bias=True)
        self.key = nn.Linear(in_features=self.d_model, out_features=self.d_V, bias=True)
        self.value = nn.Linear(in_features=self.d_model, out_features=self.d_V, bias=True)
        self.output_proj = nn.Linear(in_features=self.d_V, out_features=self.d_model, bias=False) # output projection layer

    def self_attention(self, Q: t.Tensor, K: t.Tensor, V: t.Tensor, mask: Optional[t.BoolTensor] = None) -> t.Tensor:       
        K_T = t.transpose(K, -1, -2)  # [b, N, D]
        S = t.matmul(Q, K_T) * self.scaling_factor  # attention score
        if mask is not None: S = t.masked_fill(S, mask==0, -t.inf) # mask (if any)
        A = t.softmax(S, dim=-1) # attention weight
        Z = t.matmul(A, V)  # context vector
        return Z

    def forward(self, x: t.Tensor, mask: Optional[t.BoolTensor] = None) -> t.Tensor:
        Q = self.query(x) # [b, N, D_V]
        K = self.key(x) # [b, N, D_V]
        V = self.value(x) # [b, N, D_V]
        Z = self.self_attention(Q, K, V, mask=mask) # [b, N, D_V]
        O = self.output_proj(Z) if self.projection else Z # [b, N, d_model] 
        return O



# https://arxiv.org/abs/1706.03762
class MHA(nn.Module):
    def __init__(self, d_model: int = 512, d_V: int = 64, H: int = 8) -> None: # H: total heads 
        super().__init__()
        # input features: H * d_V. output features: d_model
        self.proj = nn.Linear(in_features=H * d_V, out_features=d_model, bias=False) 
        self.multihead = nn.ModuleList([StandardAttention(d_model, d_V, False) for _ in range(H)])
    
    def forward(self, x: t.Tensor, mask: Optional[t.BoolTensor] = None) -> t.Tensor:
        Z = t.cat([head(x, mask) for head in self.multihead], dim=2) 
        O = self.proj(Z)
        return O



# https://arxiv.org/pdf/1911.02150.pdf
class MQA(StandardAttention):
    def __init__(self, d_model: int = 512, d_V: int = 64, n_queries: int = 8) -> None:
        super().__init__(d_model, d_V)
        self.n_queries = n_queries
        self.proj = nn.Linear(in_features=d_V * n_queries, out_features=d_model, bias=False) 
        delattr(self, 'query') # remove inherited query

        self.queries = nn.ModuleList([nn.Linear(in_features=d_model, out_features=d_V, bias=True) for _ in range(n_queries)])
        self.key = nn.Linear(in_features=d_model, out_features=d_V, bias=True)
        self.value = nn.Linear(in_features=d_model, out_features=d_V, bias=True)

    
    def forward(self, x: t.Tensor, mask: Optional[t.BoolTensor] = None) -> t.Tensor:
        K = self.key(x)
        V = self.value(x)
        Z = t.cat([self.self_attention(query(x), K, V, mask) for query in self.queries], dim=2)
        O = self.proj(Z)
        return O



# https://arxiv.org/pdf/2305.13245.pdf
class GQA(StandardAttention):
    def __init__(self, d_model: int = 512, d_V: int = 64, n_groups: int = 4, n_queries: int = 2) -> None: # n_queries (for each group
        super().__init__(d_model, d_V)
        delattr(self, 'query')
        delattr(self, 'key')
        delattr(self, 'value')
        self.groups = nn.ModuleList([MQA(d_model=d_model, d_V=d_V, n_queries=n_queries) for _ in range(n_groups)])
        self.proj = nn.Linear(in_features=d_model * n_groups, out_features=d_model, bias=False)

    def forward(self, x: t.Tensor, mask: Optional[t.BoolTensor] = None) -> t.Tensor:
        Z = t.cat([head(x, mask) for head in self.groups], dim=2)
        O = self.proj(Z)
        return O
