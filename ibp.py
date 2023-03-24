# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:34:09 2023

@author: marip
"""
import torch
import torch.nn as nn

class IBPLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(IBPLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.initialize_parameters()

    def forward(self, x):
        return self.linear(x)

    def interval_bound_propagation(self, l, u):
        W, b = self.linear.weight, self.linear.bias
        l_out = torch.matmul(W.clamp(min=0), l) + torch.matmul(W.clamp(max=0), u) + b
        u_out = torch.matmul(W.clamp(min=0), u) + torch.matmul(W.clamp(max=0), l) + b
        return l_out, u_out
    
    def initialize_parameters(self):
        nn.init.constant_(self.linear.weight, 0.1)
        nn.init.constant_(self.linear.bias, 0.1)

class IBPReLU(nn.Module):
    def __init__(self):
        super(IBPReLU, self).__init__()
    def forward(self, x):
        return torch.relu(x)

    def interval_bound_propagation(self, l, u):
        l_out = torch.relu(l)
        u_out = torch.relu(u)
        return l_out, u_out

class IBPModel(nn.Module):
    def __init__(self):
        super(IBPModel, self).__init__()
        self.layers = nn.Sequential(
            IBPLinear(2, 10),
            IBPReLU(),
            IBPLinear(10, 10),
            IBPReLU(),
            IBPLinear(10, 10),
            IBPReLU(),
            IBPLinear(10, 5)
        )

    def forward(self, x):
        return self.layers(x)

    def compute_bounds(self, x_bounds):
        l, u = x_bounds
        for layer in self.layers:
            l, u = layer.interval_bound_propagation(l, u)
        return l, u

# Define the IBP model
model = IBPModel()

# Input bounds: lower and upper bounds for each input feature
input_bounds = (torch.tensor([0.4, 0.4]), torch.tensor([0.6, 0.6]))

# Compute output bounds using IBP
output_bounds = model.compute_bounds(input_bounds)

# Print the output bounds
print("Output bounds:", output_bounds)