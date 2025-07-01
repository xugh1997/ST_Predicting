# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GATOperator(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, n_nodes):
        super(GATOperator, self).__init__()
        self.key_layer = nn.Linear(hidden_dims, hidden_dims)
        self.query_layer = nn.Linear(hidden_dims, hidden_dims)
        self.value_layer = nn.Linear(hidden_dims, hidden_dims)
        self.minimum = torch.full((n_nodes, n_nodes), -1e11)
        self.minimum = self.minimum.to(DEVICE)
        self.to(DEVICE)
    def forward(self, x):
        bs, n_nodes, hidden_dims = x.size()
        key = self.key_layer(x)
        query = self.query_layer(x)
        attention_scores = torch.matmul(query, key.permute(0, 2, 1))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        value = self.value_layer(x)
        out = torch.matmul(attention_probs, value)
        return F.relu((out))
class adp_hygraph_generation(nn.Module):
    def __init__(self, N_nodes, N_edges, embed_dim):
        super(adp_hygraph_generation, self).__init__()
        self.node_attr = nn.Parameter(torch.rand(N_nodes, embed_dim))
        self.edge_attr = nn.Parameter(torch.rand(N_edges, embed_dim))
        
    def forward(self):
        DE = torch.tanh(2 * self.node_attr)
        EE = torch.tanh(2 * self.edge_attr).transpose(1, 0)
        adj = F.relu(torch.tanh(2 * torch.matmul(DE, EE)))

        return adj

class BatchedHypergraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = HypergraphConv(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight, num_nodes, num_edges):
        """
        x:           (B, N, in_dim)
        edge_index:  (2, num_connections)
        edge_weight: (num_connections,)
        """
        batch_size, N, in_dim = x.shape
        x = x.reshape(batch_size * N, in_dim)  # [B*N, in_dim]

        batched_edge_index, batched_edge_weight = build_batch_edge_index_and_weight(
            edge_index, edge_weight, batch_size, num_nodes, num_edges
        )

        out = nn.functional.relu(self.conv(x, batched_edge_index, batched_edge_weight))  # [B*N, out_dim]
        return out.reshape(batch_size, N, -1)


def build_batch_edge_index_and_weight(edge_index, edge_weight, batch_size, num_nodes, num_edges):
    """
    构造 batched edge_index 和对应权重
    """
    row, col = edge_index  # shape: [num_connections]
    device = edge_index.device

    row_list = []
    col_list = []
    weight_list = []

    for b in range(batch_size):
        row_b = row + b * num_nodes
        col_b = col + b * num_edges
        row_list.append(row_b)
        col_list.append(col_b)
        weight_list.append(edge_weight.clone())  # 或者你有每个batch不同权重，这里改成传 list

    row_cat = torch.cat(row_list, dim=0)
    col_cat = torch.cat(col_list, dim=0)
    weight_cat = torch.cat(weight_list, dim=0)

    edge_index_batched = torch.stack([row_cat, col_cat], dim=0)  # shape [2, B*num_connections]
    return edge_index_batched, weight_cat

class batch_hygcn(nn.Module):
    def __init__(self, in_dims, hidden):
        super(batch_hygcn, self).__init__()
        self.hgcn = BatchedHypergraphConv(hidden, hidden)
    def forward(self,X,adj):
        bs, ts, n_node, in_dim = X.size()
        ts_list = []
        for i in range(ts):
            x = X[:,i,:,:]
            n_nodes, n_edges = adj.size()
            assert n_node==n_nodes
            edge_index = adj.nonzero(as_tuple=False).T
            edge_weight = adj[edge_index[0], edge_index[1]]
            out = self.hgcn(x, edge_index, edge_weight, n_node, n_edges)
            ts_list.append(out)
        return torch.stack(ts_list).permute(1,0,2,3).contiguous()
