# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:34:09 2023

@author: Xiangmin
"""
import torch
import torch.nn as nn
import copy

class IBPModel(nn.Module):
    def __init__(self, original_nn, alpha=0.5, testing=False, eps=0.1):
        super(IBPModel, self).__init__()
        #enable testing for fixed w and b
        self.test = testing
        self.alpha = alpha
        self.epsilon = eps
        self.copynn(original_nn)

    def copynn(self, original_nn):
        copied_layers = []
        for layer in original_nn:
            # Copy the Linear feature of the layer
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
        
    def forward(self, x):
        return self.copied_nn(x)

    def compute_bounds(self, features):
        if(len(features) == 1):
            print(features.shape)
            x_bounds = features[0]
            l = torch.full_like(x_bounds, -self.epsilon) + x_bounds
            u = torch.full_like(x_bounds, self.epsilon) + x_bounds
            for layer in self.copied_nn:
                
                if isinstance(layer, nn.Linear):
                    l, u = self.interval_bound_propagation(layer, l, u)
                elif isinstance(layer, nn.ReLU):
                    l, u = self.relu_relaxaion_approximation(l, u)

            l = l.reshape(1, 64)
            u = u.reshape(1, 64)
            return l, u
        else:
            ls = []
            us = []
            for x_bounds in features:
                print(features.shape)
                print(x_bounds.shape)
                l = torch.full_like(x_bounds, -self.epsilon) + x_bounds
                u = torch.full_like(x_bounds, self.epsilon) + x_bounds
                for layer in self.copied_nn:
                    if isinstance(layer, nn.Linear):
                        l, u = self.interval_bound_propagation(layer, l, u)
                    elif isinstance(layer, nn.ReLU):
                        l, u = self.relu_relaxaion_approximation(l, u)
                    l = l.reshape(1, 64)
                    u = u.reshape(1, 64)
                ls.append(l)
                us.append(u)

            ls = torch.concatenate(ls, axis=0)
            us = torch.concatenate(us, axis=0)
            print(ls)
            print(ls.shape)
            return ls, us


    def interval_bound_propagation(self, layer, l, u):
        W, b = layer.weight, layer.bias
        l_out = torch.matmul(W.clamp(min=0), l) + torch.matmul(W.clamp(max=0), u) + b
        u_out = torch.matmul(W.clamp(min=0), u) + torch.matmul(W.clamp(max=0), l) + b
        return l_out, u_out
    
    def relu_relaxaion_approximation(self, l, u):
        slope = (u - l) / (self.alpha * (u - l) + (1 - self.alpha) * (l + u))
        bias = l - slope * l
        l_out = slope * l + bias
        u_out = slope * u + bias
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

if __name__ == "__main__":
    mymodel = MyModel()

    # Define the IBP model
    model = IBPModel(original_nn=mymodel.layers, testing=1)
    input_tensor = torch.tensor([0.5, 0.5])
    l, u = model.compute_bounds(input_tensor)

    print("Output bounds:\n", l, u)