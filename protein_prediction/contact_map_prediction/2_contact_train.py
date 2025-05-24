from transformers import BertForMaskedLM, BertConfig, BertTokenizer
from esm.data import ESMStructuralSplitDataset
import os
import numpy as np
import torch
from einops import rearrange



esm_structural_train = ESMStructuralSplitDataset(
    split_level='superfamily', 
    cv_partition='4', 
    split='train', 
    root_path = os.path.expanduser('~/.cache/torch/data/esm'),
    download=True
)


seqs = [data['seq'] for data in esm_structural_train]
contactmap = [data['dist']<8 for data in esm_structural_train]

print(esm_structural_train[0]['dist'])
print(contactmap[0])


tokenizer = BertTokenizer.from_pretrained('mytoken')
# model = BertForMaskedLM.from_pretrained('./pretrained/', output_attentions=True)
# device = 'cuda'
# model.to(device)

from transformers import AdamW
optim = AdamW(model.parameters(), lr = 1e-5)
contact_map_layer = torch.nn.Linear(144, 1)
activation = torch.nn.Sigmoid()



def generate_batch_contact_map(seqs, contactmap, batch_size, device):
    n = len(seqs)
    nb = n//batch_size
    for i in range(nb):
        inputs = [' '.join(seq) for seq in seqs[i*batch_size:(i+1)*batch_size]]
        inputs = tokenizer(inputs, return_tensors='pt', padding=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['token_type_ids'].to(device)
        inputs['attention_mask'].to(device)
        ndim = inputs['attention_mask'].shape[1]
        maps = np.zeros((batch_size, ndim, ndim, 1))
        # 배치 단위로 cocat map에서 true r값을 넣어줌
        for j, c in enumerate(contactmap[i*batch_size:(i+1)*batch_size]):
            cdim = c.shape[0]
            # j 번째 배치의 h, w
            maps[j, 1:cdim+1, 1:cdim+1, 0] = c
        maps = torch.FloatTensor(maps)
        maps.to(device)
        yield inputs, maps
        
        
        
gen = generate_batch_contact_map()
for idx, (inputs, maps) in gen:
    break
    
    
    
epochs = 10
batch_size = 10
for epoch in range(epochs):
    gen = generate_batch_contact_map(seqs, contactmap, batch_size, device)
    for inputs, maps in gen:
        optim.zero_grad()
        out = model(**inputs)
        attn = torch.concat(out['attentions'], axis=1)
        attn = rearrange(attn, 'b n s t -> b s t n')
        contact_pred_output = activation(contact_map_layer(attn))
        
        mask = rearrange(inputs['attention_mask'], 'b n  -> b n 1 1') * rearrange(inputs['attention_mask'], 'b n  -> b 1 n 1')
        y_pred = contact_pred_output[mask>0]
        y = maps[mask>0]
        
        p = sum(y)/y.shape[0]
        w1 = 1/p
        w2 = 1/(1-p)
        weight = (w1 - w2) * y + w2
        lossfn = torch.nn.BCELoss(weight=weight)
        
        loss = lossfn(y_pred, y)
        loss.backward()
        optim.step()
        print('loss : ',loss.item())
        
        
def get_contact_prediction(model, seq, tokenizer):
    inputs = ' '.join(seq)
    inputs = tokenizer(inputs, return_tensors='pt', padding=True)
    inputs['input_ids'].to(device)
    inputs['token_type_ids'].to(device)
    inputs['attention_mask'].to(device)
    out = model(**inputs)
    attn = torch.concat(out['attentions'], axis=1)
    attn = rearrange(attn, 'b n s t -> b s t n')
    contact_pred_output = activation(contact_map_layer(attn))
    contact_pred_output = contact_pred_output[0, 1:-1, 1:-1, 0].detach().numpy()
    return contact_pred_output


seq = seqs[0]
output = get_contact_prediction(model, seq, tokenizer)


import matplotlib.pyplot as plt
plt.matshow(output)



plt.matshow(contactmap[0])