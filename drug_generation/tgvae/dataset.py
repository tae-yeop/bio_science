import os.path as op

import torch
from torch_geometric.data import Data
from tqdm import tqdm 
from rdkit import Chem 

edge_vocab = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3}


def process_smi(smi, vocab, tokenizer) : 
    out = []
    for t in tokenizer(smi) : 
        if t not in vocab : vocab[t] = len(vocab)
        out.append(vocab[t])
    out = [vocab['[START]']] + out + [vocab['[END]']]
    return torch.tensor(out, dtype=torch.long)

def process_graph(smi, graph_vocab, edge_vocab) : 
    mol = Chem.MolFromSmiles(smi)
    node_feature, edge_index, edge_attr = [], [], []

    for atom in mol.GetAtoms() : 
        symbol = atom.GetSymbol() 
        if symbol not in graph_vocab : graph_vocab[symbol] = len(graph_vocab)
        node_feature.append(graph_vocab[symbol])

    for bond in mol.GetBonds() : 
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[b, e], [e, b]]
        edge_attr += [edge_vocab[str(bond.GetBondType())]] * 2

    node_feature = torch.tensor(node_feature)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().view(2, -1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    return node_feature, edge_index, edge_attr

def process_data(data) : 
    smi_vocab = {'[START]': 0, '[END]': 1, '[PAD]': 2}
    graph_vocab = {}

    smi_list = []
    node_feature_list, edge_index_list, edge_attr_list = [], [], []

    for smi in tqdm(data, desc='Processing data') : 

        tokenized_smi = process_smi(smi, smi_vocab)
        node_feature, edge_index, edge_attr = process_graph(smi, graph_vocab, edge_vocab)

        smi_list.append(tokenized_smi)
        node_feature_list.append(node_feature)
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)

    smi_list = pad_sequence(smi_list, batch_first=True, padding_value=smi_vocab['[PAD]'])

    return smi_list, node_feature_list, edge_index_list, edge_attr_list, smi_vocab, graph_vocab, smi_list.shape[1]



def get_dataset(path_raw, path_processed) : 
    if op.exists(op.join(path_processed, 'data.pt')) :
        return load_processed_data(path_processed) 
    else : 
        raw_smi = read_smi(path_raw)
        smi, node_feature, edge_index, edge_attr, vocab_smi, vocab_graph, max_token = process_data(raw_smi)
        dataset = [MyData(x=nf, edge_index=ei,edge_attr=ea, smi=s) for nf, ei, ea, s in zip(node_feature, edge_index, edge_attr, smi)]
        save_processed_data(path_processed, dataset, vocab_smi, vocab_graph, max_token)
        return dataset, vocab_smi, vocab_graph, max_token
    

from torch_geometric.data import Dataset, Data
import re

class SMILESProcessor():
    """
    txt -> pt 파일 만들기
    """
    def __init__(self, path_raw, path_processed):
        """
        Args:
            path_raw (str): txt 경로
            path_processed (str): 경로
        """
        self.path_raw = path_raw
        self.path_processed = path_processed

    def read_smi(self, path, delimiter='\t', titleLine=False):
        result = [] 
        if path.endswith('.txt') :
            with open(path, 'r') as f : 
                for smi in tqdm(f.readlines(), desc='Reading SMILES') : 
                    if Chem.MolFromSmiles(smi) is not None : 
                        result.append(smi.strip())
        elif path.endswith('.sdf') :
            supplier = Chem.SDMolSupplier(path)
            for mol in tqdm(supplier, desc='Reading SMILES') : 
                if mol is None : 
                    continue 
                result.append(Chem.MolToSmiles(mol))
        elif path.endswith('.smi') :
            supplier = Chem.SmilesMolSupplier(path, delimiter=delimiter, titleLine=titleLine)
            for mol in tqdm(supplier, desc='Reading SMILES') : 
                if mol is None : 
                    continue 
                result.append(Chem.MolToSmiles(mol))
        return result

    def tokenize(self, smi):
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regezz = re.compile(pattern)
        tokens = [token for token in regezz.findall(smi)]
        assert smi == ''.join(tokens), ("{} could not be joined".format(smi))
        return tokens

    def smi_to_tensor(self, smi, vocab):
        out = []
        tokens self.tokenize(smi)
        for t in tokens:

    def run(self):
        list_of_smi = self.read_smi(self.path_raw)

        smi_vocab = {'[START]': 0, '[END]': 1, '[PAD]': 2}
        graph_vocab = {}
        
        smi_list = []
        node_feature_list, edge_index_list, edge_attr_list = [], [], []

        for smi in tqdm(list_of_smi, desc='Processing data') : 
            tokenized_smi = self.smi_to_tensor(smi, smi_vocab)



class SMILESDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SMILESDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['data.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.processed_paths)

if __name__ == '__main__':
    print('dsad')