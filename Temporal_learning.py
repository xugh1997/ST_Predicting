# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalCNOperator(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, ks=3):
        super(CausalCNOperator, self).__init__()
        self.ps = ks - 1
        self.causal_conv = nn.Conv1d(in_dims, hidden_dims, ks, padding=self.ps)
        self.to(DEVICE)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.causal_conv(x)
        out = F.relu(out)
        return out[:, :, :-self.ps].permute(0,2,1).contiguous()