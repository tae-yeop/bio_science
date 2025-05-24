from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
import numpy as np
from torch.distributions import Categorical

import debugpy
debugpy.listen(('0.0.0.0', 5678))
debugpy.wait_for_client()


model = AutoModelForMaskedLM.from_pretrained("TianlaiChen/PepMLM-650M")
tokenizer = AutoTokenizer.from_pretrained("TianlaiChen/PepMLM-650M")


def generate_peptide(input_seqs, peptide_length=15, top_k=3, num_binders=4):
    if isinstance(input_seqs, str):
        binders = generate_peptide_for_single_seq(input_seqs, peptide_length, top_k, num_binders)
        return pd.DataFrame(binders, columns=['Peptide', 'Pseudo Perplexity'])
    
    elif isinstance(input_seqs, list):
        results = []
        for seq in input_seqs:
            binders = generate_peptide_for_single_seq(seq, peptide_length, top_k, num_binders)
            for binder, ppl in binders:
                results.append([seq, binder, ppl])
        return pd.DataFrame(results, columns=['Input Sequence', 'Peptide', 'Pseudo Perplexity'])
    


def generate_peptide_for_single_seq(protein_seq, peptide_length=15, top_k=3, num_binders=4):
    
    # 안정성을 위해 타입 캐스팅
    peptide_length = int(peptide_length)
    top_k = int(top_k)
    num_binders = int(num_binders)

    # 최종 결과를 담을 리스트
    binders_with_ppl = []

    # 후보(binder) 수만큼 반복
    for _ in range(num_binders):
        # 1) 생성용 시퀀스 준비 
        masked_peptide = '<mask>' * peptide_length # 예: '<mask><mask>...'(15개)
        input_sequence = protein_seq + masked_peptide
        # (1, L) tensor
        inputs = tokenizer(input_sequence, return_tensors="pt").to(model.device)

        # inputs : {input_ids, attention_mask}
        # .to(model.device)는 BatchEncoding 객체 안에 든 모든 텐서를 변환

        # 2) 마스크 위치에서 토큰 예측
        with torch.no_grad():
            # (1, L, vocab_size) = 1, 175, 33
            logits = model(**inputs).logits

        # mask_token_indices.shape = torch.Size([15])
        mask_token_indices = (
            inputs['input_ids'] == tokenizer.mask_token_id # “해당 위치가 <mask>인가?” (1, L)
        ).nonzero(as_tuple=True)[1] # 마스크인 위치만
        # as_tuple=True는 차원별 인덱스를 튜플로 분리해서 리턴
        # (batch=1, seq_len=L)에 대해서 수행하므로 (batch_idx_tensor, position_idx_tensor) 
        # [1] 인덱싱을 통해 최종적으로 True(마스크)인 포지션 인덱스만 리턴, shape =(num_masks,)

        # (peptide_length, vocab)
        # 마스크인 포지션 위치의 logits만 가져옴
        logits_at_masks = logits[0, mask_token_indices]


        # 3) top‑k 필터링 & 샘플링
        # 각 펩타이드의 로짓마다 상위권 3개값을 가져옴
        top_k_logits, top_k_indices = logits_at_masks.topk(top_k, dim=-1) # 둘 다 (peptide_length, k)
        # 로짓으로 부터 확률 계산
        probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1) # (peptide_length, k)
        predicted_indices = Categorical(probabilities).sample() # (peptide_length,)
        predicted_token_ids = top_k_indices.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)

        # 4) 토큰 id → 문자열 서열 변환
        generated_binder = tokenizer.decode(predicted_token_ids, skip_special_tokens=True).replace(' ', '')


        # 5) Pseudo‑Perplexity 계산 
        ppl_value = compute_pseudo_perplexity(model, tokenizer, protein_seq, generated_binder)

        # Add the generated binder and its PPL to the results list
        binders_with_ppl.append([generated_binder, ppl_value])

    return binders_with_ppl

def compute_pseudo_perplexity(model, tokenizer, protein_seq, binder_seq):
    # 1) 전체 시퀀스 인코딩 -----------------------------------------------
    sequence = protein_seq + binder_seq
    # (1, 175)
    original_input = tokenizer.encode(sequence, return_tensors="pt").to(model.device)
    length_of_binder = len(binder_seq) # 펩타이드 길이(=마스크할 토큰 수)

    # 2) 마스크가 하나씩만 있는 배치 생성 -------------------------------
    ## Prepare a batch with each row having one masked token from the binder sequence
    masked_inputs = original_input.repeat(length_of_binder, 1) # (B=length_of_binder, L_tot)
    positions_to_mask = torch.arange(-length_of_binder - 1, -1, device=model.device) # 음수 index -> binder 위치 
    # tensor([-16, -15, -14, -13, -12, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3, -2])

    # 각 행마다 해당 binder 위치 하나만 [MASK]로 바꿈
    # [15, 175]
    masked_inputs[torch.arange(length_of_binder), positions_to_mask] = tokenizer.mask_token_id

    # 3) 라벨 텐서 준비 (loss 계산용) ------------------------------------
    # Prepare labels for the masked tokens
    # [15, 175]
    labels = torch.full_like(masked_inputs, -100) # -100은 ignore_index
    labels[torch.arange(length_of_binder), positions_to_mask] = original_input[0, positions_to_mask]  # 정답 토큰 id

    # 4) 모델 forward & 평균 loss ----------------------------------------
    # Get model predictions and calculate loss
    with torch.no_grad():
        outputs = model(masked_inputs, labels=labels) # cross‑entropy loss 반환
        loss = outputs.loss

    # Loss is already averaged by the model
    avg_loss = loss.item()
    pseudo_perplexity = np.exp(avg_loss) # ppl 정의: exp(cross‑entropy)
    return pseudo_perplexity


# Example protein sequence from UniProt (replace with an actual sequence)
protein_seq = "MSGIALSRLAQERKAWRKDHPFGFVAVPTKNPDGTMNLMNWECAIPGKKGTPWEGGLFKLRMLFKDDYPSSPPKCKFEPPLFHPNVYPSGTVCLSILEEDKDWRPAITIKQILLGIQELLNEPNIQDPAQAEAYTIYCQNRVEYEKRVRAQAKKFAPS"


# Tokenize the protein sequence for the model
inputs = tokenizer(protein_seq, return_tensors="pt")

# Inputs: {'input_ids': tensor([[ 0, 20,  8,  6, 12,  5,  4,  8, 10,  4,  5, 16,  9, 10, 15,  5, 22, 10,
#          15, 13, 21, 14, 18,  6, 18,  7,  5,  7, 14, 11, 15, 17, 14, 13,  6, 11,
#          20, 17,  4, 20, 17, 22,  9, 23,  5, 12, 14,  6, 15, 15,  6, 11, 14, 22,
#           9,  6,  6,  4, 18, 15,  4, 10, 20,  4, 18, 15, 13, 13, 19, 14,  8,  8,
#          14, 14, 15, 23, 15, 18,  9, 14, 14,  4, 18, 21, 14, 17,  7, 19, 14,  8,
#           6, 11,  7, 23,  4,  8, 12,  4,  9,  9, 13, 15, 13, 22, 10, 14,  5, 12,
#          11, 12, 15, 16, 12,  4,  4,  6, 12, 16,  9,  4,  4, 17,  9, 14, 17, 12,
#          16, 13, 14,  5, 16,  5,  9,  5, 19, 11, 12, 19, 23, 16, 17, 10,  7,  9,
#          19,  9, 15, 10,  7, 10,  5, 16,  5, 15, 15, 18,  5, 14,  8,  2]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

results_df = generate_peptide(protein_seq, peptide_length=15, top_k=3, num_binders=5)
print(results_df)

#             Binder  Pseudo Perplexity
# 0  FDEEEEPLPRLAELE           9.840302
# 1  EDEDDEPLPYALAKL           8.809409
# 2  TEEDPPLLPRYLEEE          10.831274
# 3  FEEEDPLLRRLALKE           9.714987
# 4  EQEEEYLPRLLLALE          10.397450