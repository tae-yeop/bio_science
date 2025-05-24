import math 
import copy 
import torch 
import torch.nn as nn 
import torch_geometric.nn as gnn
import torch.nn.functional as F


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class PositionwiseFeedForward(nn.Module):
    """
    Args:
        d_model: input dimension
        d_ff: feed forward dimension
        dropout: dropout rate
        x: [batch_size, seq_len, d_model]
        out: [batch_size, seq_len, d_model]
    Returns:
        out: [batch_size, seq_len, d_model]

    x → Linear(d_model, d_ff) → ReLU → dropout → Linear(d_ff, d_model) 
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.leaky_relu(self.w_1(x))))



class Embeddings(nn.Module):
    def __init__(self, d_model, size_vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(size_vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    """
    Args:
        x: [batch_size, seq_len, d_model]
    
    Returns:
        out: [batch_size, seq_len, d_model]
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # position: [max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(1)

        # div_term: [d_model // 2]
        # 값: [1, 1e-4, 1e-8, ...] 식으로 지수적으로 줄어드는 값
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # position * div_term: broadcasting으로 [max_len, d_model // 2]
        # -> sin, cos 각각 반씩 나눠서 채움
        pe[:, 0::2] = torch.sin(position * div_term) # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term) # 홀수 인덱스

        # pe: [1, max_len, d_model] → 배치 차원을 위해 unsqueeze(0)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe) # requires_grad: False

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # self.pe: [1, max_len, d_model]
        # self.pe[:, :x.size(1)] → [1, seq_len, d_model] → broadcasting됨

        # 위치 임베딩을 입력에 더함
        x = x + self.pe[:, : x.size(1)]
        # dropout 후 출력: [batch_size, seq_len, d_model]
        out = self.dropout(x)
        return out

class SmilesEmbedding(nn.Module):
    """
    입력: SMILES 토큰 ID (batch_size, seq_len)
    → Embedding: 각 토큰 ID를 (d_model)-dim 벡터로 변환
    → PositionalEncoding: 각 위치별 고유한 벡터를 더해줌
    → 출력: (batch_size, seq_len, d_model)

    "CCO" → [5, 5, 8] → 
    Embedding: [[v5], [v5], [v8]]  (v5, v8 ∈ ℝ^d)
    + PosEnc:  [[p0], [p1], [p2]]  (p0, p1, p2 ∈ ℝ^d)
    = [[v5+p0], [v5+p1], [v8+p2]]

    """
    def __init__(self, size_vocab, dim, dropout):
        super(SmilesEmbedding, self).__init__()
        self.embedding = Embeddings(dim, size_vocab)
        self.positional_encoding = PositionalEncoding(dim, dropout)
    def forward(self, x):
        return self.positional_encoding(self.embedding(x))
    


class GraphEmbedding(nn.Module) :
    def __init__(self, size_vocab, dim, dim_edge, num_head, dropout) : 
        super(GraphEmbedding, self).__init__()
        self.embedding = Embeddings(dim, size_vocab)
        self.drop = nn.Dropout(dropout)
        self.gnn = gnn.GATv2Conv(dim, dim//num_head, heads=num_head, dropout=dropout, edge_dim=dim_edge)
        self.norm = nn.BatchNorm1d(dim)
    def forward(self, x, ei, ew) : 
        return F.leaky_relu(self.norm(self.gnn(self.drop(self.embedding(x)), ei, ew)))
    


class EncoderLayer(nn.Module):
    def __init__(
            self, 
            dim, 
            dim_ff, 
            size_edge_vocab, 
            num_head, 
            dropout_encoder, 
            dropout_gat
    ):
        super().__init__()
        self.dim = dim

        self.norm1 = nn.BatchNorm1d(dim)
        self.attn1 =  gnn.GATv2Conv(
            dim, 
            dim // num_head, 
            heads=num_head, 
            dropout=dropout_gat,
            edge_dim=size_edge_vocab
        )
        self.drop1 = nn.Dropout(dropout_encoder)

        self.norm2 = nn.BatchNorm1d(dim)
        self.attn2 = gnn.GATv2Conv(
            dim, 
            dim//num_head, 
            heads=num_head, 
            dropout=dropout_gat,
            edge_dim=size_edge_vocab
        )
        self.drop2 = nn.Dropout(dropout_encoder)

        self.norm3 = nn.BatchNorm1d(dim)
        self.ff = PositionwiseFeedForward(dim, dim_ff, dropout_encoder)
        self.drop3 = nn.Dropout(dropout_encoder)

    def forward(self, node_feature, edge_index, edge_attr):
        # residual connection


        return 

class TGVAE(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()

        self.smi_embedding = SmilesEmbedding(size_smi_vocab, dim_decoder, dropout_decoder)
        self.encoder = Encoder(dim_encoder, dim_encoder_ff, size_edge_vocab, size_graph_vocab, num_encoder_head, num_encoder_layer, dropout_encoder, dropout_gat)
        self.latent_model = LatentModel(dim_encoder, dim_latent) 
        self.decoder = Decoder(dim_decoder, dim_latent, dim_decoder_ff, num_decoder_head, num_decoder_layer, dropout_decoder)
        self.generator = nn.Linear(dim_decoder, size_smi_vocab)


if __name__ == '__main__':
    # requires_grad=False인 텐서 생성
    pe = torch.randn(10, 64, requires_grad=False)

    # 슬라이싱한 텐서
    sliced = pe[:5]  # shape: (5, 64)

    # 결과 출력
    print("원본 requires_grad:", pe.requires_grad)
    print("슬라이싱 후 requires_grad:", sliced.requires_grad)

    pe = torch.randn(10, 64, requires_grad=True)
    sliced = pe[:5]

    print("원본 requires_grad:", pe.requires_grad)
    print("슬라이싱 후 requires_grad:", sliced.requires_grad)


    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            # requires_grad=False인 텐서를 버퍼로 등록
            pe = torch.randn(10, 64)  # requires_grad=False가 기본
            self.register_buffer("pe", pe)

        def forward(self):
            print("원본 requires_grad:", self.pe.requires_grad)
            
            # slicing 수행
            sliced = self.pe[:5]

            print("슬라이싱 후 requires_grad:", sliced.requires_grad)

    # 테스트 실행
    model = MyModule()
    model()

