import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool

class GNNModel(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg  

        embedding_size = cfg.embedding_size
        n_heads = cfg.attention_heads
        n_layers = cfg.layers
        dropout_rate = cfg.dropout_rate
        top_k_ratio = cfg.top_k_ratio
        top_k_every_n = cfg.top_k_every_n
        dense_neurons = cfg.dense_neurons
        edge_dim = cfg.edge_dim

        self.conv_layers = nn.ModuleList([])
        self.transf_layers = nn.ModuleList([])
        self.pool_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])

        # First Block
        self.conv1 = TransformerConv(
            self.cfg.node_feat_size,
            self.cfg.hidden_size,
            heads=self.cfg.num_heads,
            dropout=self.cfg.dropout,
            edge_dim=self.cfg.edge_feat_size,
            beta=True
        )

        self.trasnf1 = nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.cfg.hidden_size)

        # Second Block
        for i in range(self.n_layers):

        self.conv_layers.append(self.conv1)
        self.transf_layers.append(self.trasnf1)
        self.bn_layers.append(self.bn1)


    def forward(self, x, edge_attr, edge_index, batch_index):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.trasnf1(x))
