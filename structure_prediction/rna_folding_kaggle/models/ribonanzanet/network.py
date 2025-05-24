import math
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F

import matplotlib.pyplot as plt
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from dropout import *


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, attn_mask=None):
        attn = torch.matmul(q, k.transpose(2, 3))/ self.temperature
        if mask is not None:
            attn = attn+mask # this is actually the bias
        if attn_mask is not None:
            attn=attn.float().masked_fill(attn_mask == -1, float('-1e-9'))

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head*d_v, bias=False)

        self.fc = nn.Linear(n_head*d_v, d_model)

        self.attention = ScaledDotProductAttention(d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, src_mask=None):
        """
        src_mask : [B, L]
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        b, l_q, l_k, l_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_q(q).view(b, l_q, n_head, d_k)
        k = self.w_k(k).view(b, l_k, n_head, d_k)
        v = self.w_v(v).view(b, l_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None and mask.dim() == 3:   # (B, L, L)  
            mask = mask.unsqueeze(1)               # → (B, H, L, L)

        if src_mask is not None:
            src_mask[src_mask==0]=-1 # PAD → −1, 유효 토큰 1은 그대로
            src_mask = src_mask.unsqueeze(-1).float() # [B, L, 1]
            # 한쪽이라도 PAD이면 −1”이라는 마스크 효과
            attn_mask = torch.matmul(
                src_mask,
                src_mask.permute(0, 2, 1),
            ).unsqueeze(1)

            q, attn = self.attnetion(
                q,
                k,
                v,
                mask=mask,
                attn_mask=attn_mask,
            )
        else:
            q, attn = self.attnetion(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(b, l_q, -1)

        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class Outer_Product_Mean(nn.Module):
    def __init__(self, in_dim=256, dim_msa=32, pairwise_dim=64):
        super().__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)
        self.proj_down2 = nn.Linear(dim_msa ** 2, pairwise_dim)

    def forward(self, seq_rep, pair_rep=None):
        seq_rep = self.proj_down1(seq_rep)
        outer_product = torch.einsum('bid, bjc -> bijcd', seq_rep, seq_rep)
        outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')
        outer_product = self.proj_down2(outer_product)

        if pair_rep is not None:
            outer_product += pair_rep
        
        return outer_product
    

def exists(val):
    return val is not None
def default(val, d):
    return val if val is not None else d


class TriangleMultiplicativeModule(nn.Module):
    def __init__(
            self,
            *,
            dim,
            hidden_dim = None,
            mix = 'ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'ingoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'outgoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, src_mask=None):
        """
        x          # (B, L, L, C)  –  쌍(pair) 특징 행렬,  i·j 축 대칭
        src_mask   # (B, L)        –  1(valid) / 0(PAD)
        return     # (B, L, L, C)  –  같은 shape, 내용만 갱신
        """
        src_mask = src_mask.unsqueeze(-1).float() # [B, L, 1]
        mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1))
        assert mask.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)

class TriangleAttention(nn.Module):
    def __init__(self, in_dim=128, dim=32, n_heads=4, wise='row'):
        """
        Implements Triangle Attention Mechanism.
        :param in_dim: Input feature dimension.
        :param dim: Dimension of query, key, and value per head.
        :param n_heads: Number of attention heads.
        :param wise: Whether to apply row-wise or column-wise attention.
        """
        super().__init__()
        self.n_heads = n_heads
        self.wise = wise
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.linear_for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.to_out = nn.Linear(n_heads * dim, in_dim)

    def forward(self, z, src_mask):
        """
        Forward pass for TriangleAttention.
        :param z: Input tensor of shape (B, I, J, in_dim).
        :param src_mask: Source mask of shape (B, I, J).
        :return: Output tensor of shape (B, I, J, in_dim).
        """
        # pair mask
        src_mask[src_mask==0]=-1 # PAD → −1, 유효 토큰 1은 그대로
        src_mask = src_mask.unsqueeze(-1).float() # [B, L, 1]
        attn_mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1)) # (B, I, J, 1)

        wise = self.wise
        z = self.norm(z) # (B, I, J, in_dim)

        # Compute bias and gate
        gate = self.to_gate(z)  # [1] (B, I, J, in_dim)
        b = self.linear_for_pair(z)  # [5] (B, I, J, n_heads)

        # Compute Q, K, V
        q, k, v = torch.chunk(self.to_qkv(z), 3, dim=-1) # [2], [3], [4]: each (B, I, J, n_heads * dim)
        q, k, v = map(lambda x: rearrange(x, 'b i j (h d) -> b i j h d', h=self.n_heads), (q, k, v))
        # Each: (B, I, J, n_heads, dim)
        scale = q.size(-1) ** -0.5

        if wise == 'row':
            eq_attn = 'brihd,brjhd->brijh'
            eq_multi = 'brijh,brjhd->brihd'
            b = rearrange(b, 'b i j (r h)->b r i j h', r=1)
            softmax_dim = 3
            attn_mask=rearrange(attn_mask, 'b i j->b 1 i j 1')
        elif wise == 'col':
            eq_attn = 'bilhd,bjlhd->bijlh'
            eq_multi = 'bijlh,bjlhd->bilhd'
            b = rearrange(b, 'b i j (l h)->b i j l h', l=1)
            softmax_dim = 2
            attn_mask=rearrange(attn_mask, 'b i j->b i j 1 1')
        else:
            raise ValueError('wise should be col or row!')

        # Compute attention logits
        logits = (torch.einsum(eq_attn, q, k) / scale + b) # [6], [7] (B, I, J, I, n_heads) or (B, I, J, J, n_heads)
        # plt.imshow(attn_mask[0,0,:,:,0])
        # plt.show()
        # exit()
        logits = logits.masked_fill(attn_mask == -1, float('-1e-9'))

        # Compute attention weights
        attn = logits.softmax(softmax_dim) # [8] (B, I, J, I, n_heads) or (B, I, J, J, n_heads)
        # print(attn.shape)
        # print(v.shape)
        out = torch.einsum(eq_multi, attn, v) # [9] (B, I, J, n_heads, dim)
        out = gate * rearrange(out, 'b i j h d-> b i j (h d)') # [10] (B, I, J, in_dim)
        # Final projection
        z_ = self.to_out(out) # (B, I, J, in_dim)
        return z_


class ConvTransformerEncoderLayer(nn.Module):
    """
    A Transformer Encoder Layer with convolutional enhancements and pairwise feature processing.
    """
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        pairwise_dimension, 
        use_triangular_attention, 
        dropout=0.1, k = 3,
    ):
        """
        :param d_model: Dimension of the input embeddings
        :param nhead: Number of attention heads
        :param dim_feedforward: Hidden layer size in feedforward network
        :param pairwise_dimension: Dimension of pairwise features
        :param use_triangular_attention: Whether to use triangular attention modules
        :param dropout: Dropout rate
        :param k: Kernel size for the 1D convolution
        """
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.pairwise2heads=nn.Linear(pairwise_dimension,nhead,bias=False)
        self.pairwise_norm=nn.LayerNorm(pairwise_dimension)
        self.activation = nn.GELU()

        self.conv = nn.Conv1d(d_model, d_model, k, padding=k//2)

        self.triangle_update_out=TriangleMultiplicativeModule(dim=pairwise_dimension,mix='outgoing')
        self.triangle_update_in=TriangleMultiplicativeModule(dim=pairwise_dimension,mix='ingoing')

        self.pair_dropout_out=DropoutRowwise(dropout)
        self.pair_dropout_in=DropoutRowwise(dropout)

        self.use_triangular_attention=use_triangular_attention
        if self.use_triangular_attention:
            self.triangle_attention_out=TriangleAttention(in_dim=pairwise_dimension,
                                                                    dim=pairwise_dimension//4,
                                                                    wise='row')
            self.triangle_attention_in=TriangleAttention(in_dim=pairwise_dimension,
                                                                    dim=pairwise_dimension//4,
                                                                    wise='col')

            self.pair_attention_dropout_out=DropoutRowwise(dropout)
            self.pair_attention_dropout_in=DropoutColumnwise(dropout)

        self.outer_product_mean=Outer_Product_Mean(in_dim=d_model,pairwise_dim=pairwise_dimension)

        self.pair_transition=nn.Sequential(
                                           nn.LayerNorm(pairwise_dimension),
                                           nn.Linear(pairwise_dimension,pairwise_dimension*4),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pairwise_dimension*4,pairwise_dimension))

    def forward(self, src , pairwise_features, src_mask=None, return_aw=False):
        """
        Forward pass of the ConvTransformerEncoderLayer.

        :param src: Input tensor of shape (batch_size, seq_len, d_model)
        :param pairwise_features: Pairwise feature tensor of shape (batch_size, seq_len, seq_len, pairwise_dimension)
        :param src_mask: Optional mask tensor of shape (batch_size, seq_len)
        :param return_aw: Whether to return attention weights
        :return: Tuple containing processed src and pairwise_features (and optionally attention weights)
        """
        src = src * src_mask.float().unsqueeze(-1) # Shape: (batch_size, seq_len, d_model)

        res = src

        src = src + self.conv(src.permute(0,2,1)).permute(0,2,1) # Shape: (batch_size, seq_len, d_model)
        src = self.norm3(src)

        pairwise_bias = self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(0,3,1,2)
        src2, attention_weights = self.self_attn(src, src, src, mask=pairwise_bias, src_mask=src_mask)  # Shape: (batch_size, seq_len, d_model)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))) # Shape: (batch_size, seq_len, d_model)

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        pairwise_features += self.outer_product_mean(src) # Shape: (batch_size, seq_len, seq_len, pairwise_dimension)
        pairwise_features += self.pair_dropout_out(self.triangle_update_out(pairwise_features, src_mask))
        pairwise_features += self.pair_droptout_in(self.triangle_update_in(pairwise_features, src_mask))

        if self.use_triangular_attention:
            pairwise_features += self.pair_attention_dropout_out(self.triangular_attention_out(pairwise_features, src_mask))
            pairwise_features += self.pair_attention_dropout_in(self.triangular_attention_in(pairwise_features, src_mask))

        pairwise_features += self.pair_transition(pairwise_features) # Shape: (batch_size, seq_len, seq_len, pairwise_dimension)

        if return_aw:
            return src, pairwise_features, attention_weights # Shapes: (batch_size, seq_len, d_model), (batch_size, seq_len, seq_len, pairwise_dimension), (batch_size, nhead, seq_len, seq_len)
        else:
            return src, pairwise_features # Shapes: (batch_size, seq_len, d_model), (batch_size, seq_len, seq_len, pairwise_dimension)


class RelativePositionalEncoding(nn.Module):
    """
    Implements relative positional encoding for sequence-based models.
    :param dim: (int) The output embedding dimension. Default is 64.
    """
    def __init__(self, dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(17, dim)  # (17,) -> (dim,)

    def forward(self, src: torch.Tensor):
        """
        Computes the relative positional encodings for a given sequence.

        :param src: Input tensor of shape (B, L, D), where:
            - B: Batch size
            - L: Sequence length
            - D: Feature dimension (ignored in this module)
        :return: Relative positional encoding of shape (L, L, dim)
        """
        L = src.shape[1]  # Sequence length
        res_id = torch.arange(L, device=src.device).unsqueeze(0)  # (1, L)

        device = res_id.device
        bin_values = torch.arange(-8, 9, device=device)  # (17,)

        d = res_id[:, :, None] - res_id[:, None, :]  # (1, L, L)

        bdy = torch.tensor(8, device=device)

        # Clipping the values within the range [-8, 8]
        d = torch.minimum(torch.maximum(-bdy, d), bdy)  # (1, L, L)

        # One-hot encoding of relative positions
        d_onehot = (d[..., None] == bin_values).float()  # (1, L, L, 17)

        assert d_onehot.sum(dim=-1).min() == 1  # Ensure proper one-hot encoding

        # Linear transformation to embedding space
        p = self.linear(d_onehot)  # (1, L, L, 17) -> (1, L, L, dim)

        return p.squeeze(0)  # (L, L, dim)


class RibonanzaNet(nn.Module):
    def __init__(self, config: object):
        """
        Initializes the RibonanzaNet model.
        
        :param config: Configuration object containing model hyperparameters.
            - ninp (int): Input embedding dimension.
            - ntoken (int): Vocabulary size for embedding layer.
            - nclass (int): Number of output classes.
            - nhead (int): Number of attention heads.
            - nlayers (int): Number of transformer encoder layers.
            - dropout (float): Dropout probability.
            - pairwise_dimension (int): Dimension of pairwise features.
            - use_triangular_attention (bool): Whether to use triangular attention.
            - use_bpp (bool): Whether to use base-pairing probability features.
            - k (int): Kernel size for convolutions in transformer layers.
        """
        super().__init__()
        self.config = config
        nhid = config.ninp * 4
    
        self.transformer_encoder = []
        print(f"Constructing {config.nlayers} ConvTransformerEncoderLayers")

        for i in range(config.nlayers):
            k = config.k if i != config.nlayers - 1 else 1
            self.transformer_encoder.append(
                ConvTransformerEncoderLayer(
                    d_model=config.ninp, nhead=config.nhead,
                    dim_feedforward=nhid,
                    pairwise_dimension=config.pairwise_dimension,
                    use_triangular_attention=config.use_triangular_attention,
                    dropout=config.dropout, k=k)
            )   
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)

        self.encoder = nn.Embedding(config.ntoken, config.ninp, padding_idx=4)
        self.decoder = nn.Linear(config.ninp, config.nclass)

        if config.use_bpp:
            self.mask_dense = nn.Conv2d(2, config.nhead // 4, 1)
        else:
            self.mask_dense = nn.Conv2d(1, config.nhead // 4, 1)

        self.OuterProductMean = OuterProductMean(in_dim=config.ninp, pairwise_dim=config.pairwise_dimension)
        self.pos_encoder = RelativePositionalEncoding(config.pairwise_dimension)


    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: torch.Tensor = None, 
        return_aw: bool = False
    ):
        """
        Forward pass of the RibonanzaNet model.
        
        :param src: Input tensor of shape (B, L), where B is the batch size and L is the sequence length.
        :param src_mask: Optional mask tensor of shape (B, L, L), used for attention masking.
        :param return_aw: Boolean flag indicating whether to return attention weights.
        :return: Output tensor of shape (B, L, nclass) if return_aw is False, or a tuple (output, attention_weights).
        """
        B, L = src.shape  # (Batch size, Sequence length)
        src = self.encoder(src).reshape(B, L, -1)  # (B, L, ninp)

        pairwise_features = self.OuterProductMean(src)  # (B, L, L, pairwise_dimension)
        pairwise_features = pairwise_features + self.pos_encoder(src)  # (B, L, L, pairwise_dimension)

        attention_weights = []
        for i, layer in enumerate(self.transformer_encoder):
            if src_mask is not None:
                if return_aw:
                    src, aw = layer(src, pairwise_features, src_mask, return_aw=return_aw)
                    attention_weights.append(aw)
                else:
                    src, pairwise_features = layer(src, pairwise_features, src_mask, return_aw=return_aw)
            else:
                if return_aw:
                    src, aw = layer(src, pairwise_features, return_aw=return_aw)
                    attention_weights.append(aw)
                else:
                    src, pairwise_features = layer(src, pairwise_features, return_aw=return_aw)
        
        output = self.decoder(src).squeeze(-1) + pairwise_features.mean() * 0  # (B, L, nclass)

        if return_aw:
            return output, attention_weights
        else:
            return output

        

if __name__ == '__main__':
    ...