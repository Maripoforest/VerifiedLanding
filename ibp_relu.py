# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:34:09 2023

@author: marip
"""
import torch
import torch.nn as nn
import copy

class IBPModel(nn.Module):
    def __init__(self, original_nn, alpha=0.5, testing=False):
        super(IBPModel, self).__init__()
        self.test = testing
        self.nn = self.copynn(original_nn)
        self.alpha = alpha

    def copynn(self, original_nn):
        copied_layers = []
        for layer in original_nn:
            if isinstance(layer, nn.Linear):
                copied_layer = nn.Linear(layer.in_features, layer.out_features)
                copied_layer.weight = nn.Parameter(copy.deepcopy(layer.weight))
                copied_layer.bias = nn.Parameter(copy.deepcopy(layer.bias))
                if self.test:
                    copied_layer = self.initialize_parameters(copied_layer)
            else:
                copied_layer = copy.deepcopy(layer)
            copied_layers.append(copied_layer)
        self.copied_nn = nn.Sequential(*copied_layers)
        return copied_layers

    def forward(self, x):
        return self.copied_nn(x)

    def compute_bounds(self, x_bounds):
        l, u = x_bounds
        for layer in self.nn:
            if isinstance(layer, nn.Linear):
                l, u = self.interval_bound_propagation(layer, l, u)
            elif isinstance(layer, nn.ReLU):
                l, u = self.relu_relaxaion_approximation(l, u)
        return l, u

    def interval_bound_propagation(self, layer, l, u):
        W, b = layer.weight, layer.bias
        l_out = torch.matmul(W.clamp(min=0), l) + torch.matmul(W.clamp(max=0), u) + b
        u_out = torch.matmul(W.clamp(min=0), u) + torch.matmul(W.clamp(max=0), l) + b
        return l_out, u_out
    
    def relu_relaxaion_approximation(self, l, u):
        slope = (u - l) / (self.alpha * (u - l) + (1 - self.alpha) * (l + u))
        bias = l - slope * l
        u_out = slope * l + bias
        l_out = slope * u + bias
        return l_out, u_out
    
    def initialize_parameters(self, layer):
        nn.init.constant_(layer.weight, 0.25)
        nn.init.constant_(layer.bias, 0.05)
        return layer

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )

    def forward(self, x):
        return self.layers(x)

mymodel = MyModel()

# Define the IBP model
model = IBPModel(original_nn=mymodel.layers, testing=True)

# Input bounds: lower and upper bounds for each input feature
input_bounds = (torch.tensor([0.4, 0.4]), torch.tensor([0.6, 0.6]))

# Compute output bounds using IBP
output_bounds = model.compute_bounds(input_bounds)

# Print the output bounds
print("Output bounds:", output_bounds)