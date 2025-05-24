import torch
from packaging import version


def subsequent_mask(size):
    """
    디코더에서 현재 토큰 이후(미래)의 토큰을 attend하지 못하게 막는 마스크 생성

    Args:
        size (int): 시퀀스 길이
    Returns:
        torch.Tensor: 마스크 텐서

    [[[1, 0, 0, 0],
      [1, 1, 0, 0],
      [1, 1, 1, 0],
      [1, 1, 1, 1]]]  ← True = attend 가능
    """

    """
    Transformer 디코더용 현재 토큰 이후(미래)의 토큰을 attend하지 못하게 하는 causal mask 생성 (자기회귀 마스크).
    PyTorch 버전에 따라 bool 또는 uint8 dtype을 사용.

    Args:
        size (int): 시퀀스 길이

    Returns:
        mask (Tensor): shape = (1, size, size), dtype = bool
    """


    attn_shape = (1, size, size)
    if version.parse(torch.__version__) >= version.parse("1.2.0"):
        # 최신 버전은 bool dtype 사용
        subsequent_mask = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
        return ~subsequent_mask
    else:
        # 오래된 버전은 uint8 사용
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return subsequent_mask == 0

def get_mask(target, smi_vocab) :
    """
    [PAD] 위치를 마스킹하고, future 마스크(subsequent_mask)와 결합
    target shape: (batch_size, seq_len)
    mask shape: (batch_size, 1, seq_len)
    output shape: (batch_size, seq_len, seq_len)
    
    Args:
        target (Tensor): [batch, seq_len] 크기의 SMILES 시퀀스 텐서
        smi_vocab (dict): '[PAD]' 토큰의 인덱스를 포함한 vocab 딕셔너리
    Returns:
        mask (Tensor): [batch, seq_len, seq_len] 크기의 마스크 텐서
    """
    """

    예: batch_size = 2, seq_len = 5
    - target: (2, 5)
    - mask: (2, 1, 5)
    - subsequent_mask: (1, 5, 5)
    => broadcast & AND 연산 → 최종 마스크: (2, 5, 5)
    """
    mask = (target != smi_vocab['[PAD]']).unsqueeze(-2)
    return mask & subsequent_mask(target.size(-1)).type_as(mask.data)
def convert_data(data, smi_vocab, device=None):
    """
    데이터를 모델 입력에 맞게 변환 (graph + SMILES sequence 분리, 마스킹 포함)

    Args:
        data (Data): PyTorch Geometric Data 객체
        smi_vocab (dict): SMILES vocab 딕셔너리
        device (torch.device, optional): 사용할 디바이스. 기본값은 None (CPU)
    
    Returns:
        - inp_graph: 분자 그래프 입력
        - inp_smi: SMILES 입력 시퀀스 (decoder input, e.g., "C=CC" → "C=CC")
        - inp_smi_mask: attention mask for SMILES
        - tgt_smi: 다음 토큰을 예측하기 위한 타겟 시퀀스 (e.g., "C=CC" → "=CCC")
    """
    inp_graph, smi = data.to(device), data.smi.to(device) 
    # data.smi shape: (batch_size, seq_len)
    # inp_graph : PyG Data 또는 (batch_size, num_nodes, features)
    inp_smi, tgt_smi = smi[:, :-1], smi[:, 1:] # teacher forcing (batch_size, seq_len - 1)
    inp_smi_mask = get_mask(inp_smi, smi_vocab) # (batch_size, seq_len - 1, seq_len - 1)
    return inp_graph, inp_smi, inp_smi_mask.to(device), tgt_smi
