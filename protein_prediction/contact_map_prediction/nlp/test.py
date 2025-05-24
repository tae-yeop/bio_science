from transformers import BertForMaskedLM, BertConfig, BertTokenizer
from esm.data import ESMStructuralSplitDataset
import os
import numpy as np

esm_structural_train = ESMStructuralSplitDataset(
    split_level='superfamily', 
    cv_partition='4', 
    split='train', 
    root_path = os.path.expanduser('~/.cache/torch/data/esm'), # /home/tyk/.cache/torch/data/esm
    download=True
)

print(type(esm_structural_train))
seqs = [data['seq'] for data in esm_structural_train]


tokenizer = BertTokenizer.from_pretrained('mytoken')

configuration = BertConfig()

model = BertForMaskedLM(configuration)

configuration