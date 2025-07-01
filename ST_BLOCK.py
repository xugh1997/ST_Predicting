# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from Temporal_learning import CausalCNOperator
from Spatial_learning import GATOperator, adp_hygraph_generation, batch_hygcn
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SerialBlock(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, n_nodes):
        super(SerialBlock, self).__init__()
        self.hidden_dims = hidden_dims
        self.temporal_op1 = CausalCNOperator(DEVICE, in_dims, hidden_dims)
        self.hgcn_op_prior1 = batch_hygcn(1, hidden_dims)
        self.adphgcn1 = adp_hygraph_generation(n_nodes, int(n_nodes*0.4), 40)
        self.gat_op_data1 = GATOperator(DEVICE, in_dims, hidden_dims, n_nodes)
        self.fusion_linear1 = nn.Linear(2*hidden_dims, hidden_dims)
        self.temporal_op2 = CausalCNOperator(DEVICE, hidden_dims, hidden_dims)
        self.temporal_op3 = CausalCNOperator(DEVICE, hidden_dims, hidden_dims)
        self.hgcn_op_prior2 = batch_hygcn(hidden_dims, hidden_dims)
        self.adphgcn2 = adp_hygraph_generation(n_nodes, int(n_nodes*0.4), 40)
        self.gat_op_data2 = GATOperator(DEVICE, in_dims, hidden_dims,  n_nodes)
        self.fusion_linear2 = nn.Linear(2 * hidden_dims, hidden_dims)
        self.temporal_op4 = CausalCNOperator(DEVICE, hidden_dims, hidden_dims)
        self.linear = nn.Linear(hidden_dims, 1)

        self.to(DEVICE)
    def forward(self, x):
        bs, n_timesteps, n_nodes, in_dims = x.size()
        out =self.temporal_op1(x.permute(0,2,1,3).contiguous().reshape(bs*n_nodes,n_timesteps,in_dims))
        # x_residual2 = torch.clone(x)
        out_reshaped = out.reshape(bs, n_nodes,n_timesteps,self.hidden_dims).permute(0,2,1,3).contiguous()

        x_residual1 = torch.clone(out_reshaped)
        x_residual2 = torch.clone(out_reshaped)
        data_out1 = self.gat_op_data1(out_reshaped.reshape(bs*n_timesteps,n_nodes,self.hidden_dims)).reshape(bs, n_timesteps, n_nodes, self.hidden_dims)
        data_out1 = data_out1+x_residual1

        # data_residual = self.res_linear2(x_residual2)
        # data_out = x_data + data_residual
        adj_hgcn1 = self.adphgcn1()
        prior_out1 = self.hgcn_op_prior1(out_reshaped, adj_hgcn1)
        # prior_out = self.res_linear1(x_residual1)+hgcn_out
        prior_out1 = prior_out1+x_residual2
        out = F.relu((self.fusion_linear1(torch.concat([prior_out1, data_out1], dim=-1))))
        # x = self.start_linear(x)
        
        out2 = self.temporal_op2(out.permute(0,2,1,3).contiguous().reshape(bs*n_nodes,n_timesteps,self.hidden_dims))
        out3 = self.temporal_op3(out2)
        out_reshaped2 = out3.reshape(bs, n_nodes, n_timesteps, self.hidden_dims).permute(0, 2, 1, 3).contiguous()
        x_residual3 = torch.clone(out_reshaped2)
        x_residual4 = torch.clone(out_reshaped2)
        data_out2 = self.gat_op_data2(out_reshaped2.reshape(bs * n_timesteps, n_nodes, self.hidden_dims)).reshape(bs,n_timesteps,n_nodes,self.hidden_dims)
        data_out2 = data_out2+x_residual3
        adj_hgcn2 = self.adphgcn2()
        prior_out2 = self.hgcn_op_prior2(out_reshaped2, adj_hgcn2)
        prior_out2 = prior_out2+x_residual4
        out2 = F.relu(self.fusion_linear2(torch.concat([prior_out2, data_out2], dim=-1)))
        output = self.temporal_op4(out2.permute(0,2,1,3).contiguous().reshape(bs*n_nodes,n_timesteps,self.hidden_dims))
        output = output.reshape(bs, n_nodes, n_timesteps, self.hidden_dims)

        return nn.functional.relu(self.linear(output[:, :, -1, :]))
