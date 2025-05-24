import torch
from torch.utils.data import Dataset
from torch.nn.utils import rnn

import pandas as pd


class StringDataset(Dataset):
    def __init__(self, tokenizer, data_list):
        self.tokenizer = tokenizer
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Tokenization을 한 뒤에 EOS token을 추가
        """
        toks = self.tokenizer.tokenize(self.data[idx]) + [self.tokenizer.id2tok[self.tokenizer.get_eosi()]]
        return torch.tensor(self.tokenizer.encode(toks))
    
    def collate_fn(self, batch):
        string_ids_tensor = rnn.pad_sequence(
            batch,
            batch_first=True,
            padding_value=self.tokenizer.get_padi()
        )
        return string_ids_tensor