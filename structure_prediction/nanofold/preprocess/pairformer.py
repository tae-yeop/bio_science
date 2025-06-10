import torch
import torch.nn as nn


"""
Input:
seq_rep (MSA representation) : [B, S, L, C]
- S : MSA안에 포함된 서열 수
- L : 단백질 길이 수


pair_rep (Pair representation) : [B, L, L, C_p]
- C_p : residue-pair feature(잔기 짝 특징) 차원 = 상대거리 bucket, coevolution 정보, 템플릿 distogram 등
"""

class PairformerBlock(nn.Module):
    def __init__(
        self, 
        si

    ):

    def forward(self, single_rep, pair_rep):
        # pair rep 계산
        pair_rep = pair_rep + self.pari_d


