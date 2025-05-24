from transformers import BertForMaskedLM, BertConfig, BertTokenizer
from esm.data import ESMStructuralSplitDataset
import os
import numpy as np


esm_structural_train = ESMStructuralSplitDataset(
    split_level='superfamily', 
    cv_partition='4', 
    split='train', 
    root_path = os.path.expanduser('~/.cache/torch/data/esm'),
    download=True
)


seqs = [data['seq'] for data in esm_structural_train]

tokenizer = BertTokenizer.from_pretrained('mytoken')

configuration = BertConfig()
configuration.max_position_embeddings = 1000

model = BertForMaskedLM(configuration)

def make_input_n_label(seq):
    """
    'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'

    원본에 [MASK]와 랜덤 값이 들어간 시퀀스 그리고
    해당 [MASK] 부분에 정답이 들어간 시퀀스 (정답에 대한 아미노산), 나머지는 예측하지 말도록
    """
    seq = list(seq) # 캐릭터단위로 끊김
    seq = np.array(seq, dtype='<U6') # numpy로 바굼
    mask1 = np.random.random(size=seq.shape)<0.15 # True, False 넘파이 에러이
    u = np.random.random(size=seq.shape)
    mask = mask1 & (u < 0.8) # 공통으로 True인것만 남김
    tokens = list('ABCDEFGHIKLMNOPQRSTUVWXYZ-.')
    # 랜덤 마스크는 왜 넣는지?
    random_another_mask = mask1 & (u>=0.8) & (u<0.9)
    random_another = np.random.choice(tokens, size = seq.shape) # seq 길이만큼 랜덤 시퀀스 만듬

    seqm = seq.copy()
    seqm[mask] = '[MASK]' # 마스크 걸리는 위치에 적용
    seqm[random_another_mask] = random_another[random_another_mask] # 랜덤값 넣음
    seqL = seq.copy()
    # 반대로 원래 마스크 아닌 부분에 마스크를 넣음
    seqL[~mask] = '[MASK]'

    return ' '.join(seqm), ' '.join(seqL)

def generate_batch(seqs, batch_size, device):
    n = len(seqs)
    nb = n//batch_size
    for i in range(nb):
        example =[make_input_n_label(seq) for seq in seqs[i*batch_size:(i+1)*batch_size]]
        inputs = [e[0] for e in example]
        labels = [e[1] for e in example]
        inputs = tokenizer(inputs, return_tensors='pt', padding=True)
        labels = tokenizer(labels, return_tensors='pt', padding=True)
        # print(device)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        
        labels['input_ids'][labels['input_ids']<5]=-100
        labels['input_ids'] = labels['input_ids'].to(device)
        yield inputs, labels

from transformers import AdamW
optim = AdamW(model.parameters(), lr=1e-5)

device = 'cuda'
model = model.to(device)

epochs = 2
batch_size = 8
for epcoh in range(epochs):
    gen = generate_batch(seqs, batch_size, device)
    for idx, (inputs, labels) in enumerate(gen):
        optim.zero_grad()
        out = model(**inputs, labels=labels['input_ids'])
        loss = out.loss
        loss.backward()
        optim.step()
        if idx % 100 == 0:
            print('loss : ',loss.item())
            
            
os.makedirs('pretrained', exist_ok=True)
model.save_pretrained('pretrained') # config와 .bin이 저장된다