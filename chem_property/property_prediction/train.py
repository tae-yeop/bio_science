# https://dacon.io/competitions/official/236127/codeshare/8802?page=1&dtype=recent
import pandas as pd
train_df = pd.read_csv('/purestorage/project/tyk/14_Classification/mol/train.csv')

print(train_df.head())

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole

mol = Chem.MolFromSmiles(train_df['SMILES'][1000])

import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.utils import from_smiles

from omegaconf import OmegaConf

import debugpy

debugpy.listen(('0.0.0.0', 5678))

print("Waiting for debugger attach")
debugpy.wait_for_client()

class MultiDataset(Dataset):
    def __init__(self, tabular_path):
        super().__init__()

        self.train_df = pd.read_csv(tabular_path)
        self.train_df.fillna(0, inplace=True)
        self.graph_list = self.smiles2mol(self.train_df['SMILES'])

        self.target_mlm = torch.tensor(self.train_df['MLM'].values.astype(np.float32))
        self.target_hlm = torch.tensor(self.train_df['HLM'].values.astype(np.float32))

        self.wo_smiles_df = self.train_df.drop(columns=['SMILES', 'id', 'MLM', 'HLM'])
        self.wo_smiles_df = torch.tensor(self.wo_smiles_df.values.astype(np.float32))

    def smiles2mol(self, smiles_list):
        graph_list = []
        for smiles in smiles_list:
            graph_data = from_smiles(smiles) # 'CCOc1ccc(CNC(=O)c2cc(-c3sc(C)nc3C)n[nH]2)cc1OCC' str
            # mol = Chem.MolFromSmiles(smiles)
            # for atom in mol.GetAtoms() : ... x = torch.tensor(xs, dtype=torch.long).view(-1, 9)
            # for bond in mol.GetBonds() : ... edge_index = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
            graph_data.smiles = None
            graph_data.edge_attr = None

            graph_list.append(graph_data)

        return graph_list

    def __getitem__(self, idx):
        return self.graph_list[idx], self.wo_smiles_df[idx], self.target_mlm[idx], self.target_hlm[idx]

    def __len__(self):
        return len(self.graph_list)

# dataset = MultiDataset('/purestorage/project/tyk/14_Classification/mol/train.csv')
# d = next(iter(dataset))
# print(d)

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GraphFeature(nn.Module):
    def __init__(self, node_feat, embed_dim):
        super().__init__()
        self.conv_l1 = GCNConv(node_feat, 8)
        self.conv_l2 = GCNConv(8, 16)
        self.embedding = nn.Linear(16, embed_dim)

    def forward(self, x, edge_idx, batch):
        x = F.elu(self.conv_l1(x, edge_idx))
        x = F.elu(self.conv_l2(x, edge_idx))

        x = global_mean_pool(x, batch)

        x = self.embedding(x)
        return x

from pytorch_tabnet.tab_network import AttentiveTransformer, FeatTransformer
class TabNetEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=None,
        device='cpu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = nn.BatchNorm1d(self.input_dim, momentum=0.01)
        self.group_attention_matrix = group_attention_matrix
        self.device = device

        if self.group_attention_matrix is None:
            # no groups
            self.group_attention_matrix = torch.eye(self.input_dim).to(self.device)
            self.attention_dim = self.input_dim
        else:
            self.attention_dim = self.group_attention_matrix.shape[0]

        if self.n_shared > 0:

        else:
             shared_feat_transform = None

         self.initial_splitter = FeatTransformer(
            self.input_dim,
            n_d + n_a,
            shared_feat_transform,

         )

    def forward(self, x, prior=None):
        x = self.initial_bn(x)

        bs = x.shape[0]  # batch size
        if prior is None:
            prior = torch.ones((bs, self.attention_dim), device=x.device)

        M_loss = 0
        att = ...


class GraphTab(nn.Module):
    def __init__(
        self, 
        cfgs,
        num_heads,

    ):
        super().__init__()

        self.graph_feature = GraphFeature()
        self.tabnet_feature = TabNetEncoder()
        self.attn_layer = nn.MultiheadAttention()
        self.regressor = nn.Sequnetial(
            nn.
        )

    def forward(self, node_attr, edge_idx, batch, tabular):
        gr_ft = self.graph_feature(node_attr, edge_idx, batch) # (batch, embed_dim)
        tab_ft = self.tabnet_feature(tabular) #
        tab_ft = torch.sum(torch.stack(tab_ft[0]), dim=0)

        # 필요한가?
        # gr_ft.to(self.device)
        # tab_ft.to(self.device)
        
        # q, k, v 순인데 이게 맞을까?
        attn_output = self.attn_layer(gr_ft, gr_ft, tab_ft)[0] # (batch, pos_emb)
        # 실제 attention output. tab_ft와 같은 shape가 아닐까?

        res = self.regressor(attn_output)
        return res

if __name__ == '__main__':
    cfgs = {
    'node_feat': 9,
    'embed_dim': 32,
    'input_dim': 7,
    'output_dim': 0,
    'n_da': 32, # output dimension
    'n_steps': 3,
    'num_heads': ,
    # training
    'epcohs': 100
    }

    cfgs = OmegaConf.load(cfgs)
    OmegaConf.set_struct(cfgs, False)

    trainset = MultiDataset('/purestorage/project/tyk/14_Classification/mol/train.csv')
    train_loader = DataLoader(trainset, batch_size=128)

    model = GraphTab(
        cfgs,
        num_heads=4,
        reg_emb=32,
        drop_ratio=0.1,
        out_dim=1,
    ).to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    losses = []

    model.train()
    for epoch in range(cfgs.epochs):
        for gr, tr_df, mlm_target, _ in train_loader:
            gr_x = gr.x.type(torch.float32).to('cuda')
            gr_edge_idx = gr.edge_index.to('cuda')
            gr_batch = gr.batch.to('cuda')

            tabular = tr_df.to('cuda')
            mlm_target = mlm_target.view(-1, 1).to('cuda')

            optimizer.zero_grad()
            predict = model(gr_x, gr_edge_idx, gr_batch, tabular)

            loss = criterion(predict, mlm_target)
            loss.backward()
            ep_loss = loss.item()
            optimizer.step()
        losses.append(ep_loss)
        print(f'Epoch: {ep+1} / rmse: {ep_loss}')
    


