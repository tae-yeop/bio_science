import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import MolToSmiles


class MolecularDataset(Dataset):
    def __init__(
            self,
            csv_path,
            tokenizer_path, 
            dataset_type='pdbbind', 
            max_length=512
    ):  
        super().__init__()

        self.data = pd.read_csv(csv_path)
        self.dataset_type = dataset_type
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length

