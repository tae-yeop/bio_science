import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearWithView(nn.Module):
    def __init__(self, in_dim, out_features, *args, **kwargs):
        super().__init__()
        self.out_features = out_features
        self.linear = nn.Linear(in_dim, math.prod(out_features), *args, **kwargs)

    def forward(self, x):
        out = self.linear(x)
        return out.view(*out.shape[:-1], *self.out_features)


class AttentionWithPairBias(nn.Module):
    def __init__(self, in_dim, num_heads, head_dim):
        super().__init__()

        self.q = LinearWithView(in_dim, (num_heads, head_dim), bias=False)
        self.k = LinearWithView(in_dim, (num_heads, head_dim), bias=False)
        self.v = LinearWithView(in_dim, (num_heads, head_dim), bias=False)

        self.pair_bias = LinearWithView(in_dim, head_dim, bias=False)

        self.bias = nn.Sequential(
            nn.LayerNorm(pa),
            nn.Lnear()
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, num_heads * head_dim, bias=False),
            nn.Sigmoid()
        )
        
        self.out = nn.Linear(num_heads * head_dim, in_dim, bias=False)
    

    def forward(self, single_rep, pair_rep):

        # [B, L, D]
        single_rep = self.layer_norm(single_rep)

        query = self.q(single_rep) # [B, L, num_heads, head_dim]
        key = self.k(single_rep)
        value = self.v(single_rep)

        pair_rep = self.pair_bias(pair_rep) # [B, L, L, head_dim]]
        
        # [B, H, L, HD]
        attn = F.scaled_dot_product_attention(
            query.transpose(-3, -2),
            key.transpose(-3, -2),
            value.transpose(-3, -2),
            bias=pair_rep.movedim()
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )
        
        attn = attn.transpose(-3, -2).view(*attn.shape[:-2], -1)  # [B, L, num_heads * head_dim]

        gate = self.gate(single_rep)  # [B, L, num_heads * head_dim]
        attn = attn * gate

        out = self.out(attn)  # [B, L, D]
        return out



