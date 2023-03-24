# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:57:08 2023

@author: marip
"""

import torch
import torch.nn as nn

class CrownModel(nn.Module):
    def __init__(self):
        super(CrownModel, self).__init__()
        # Define your neural network architecture
        self.layers = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def backward_bound_propagation(self, output_bounds):
        # Initialize the bounds for the output layer
        pre_bounds, post_bounds = output_bounds

        # Iterate through the layers in reverse order
        for layer in reversed(self.layers):
            if isinstance(layer, nn.Linear):
                # Compute the pre-activation bounds for the linear layer
                pre_bounds = compute_linear_bounds(layer, post_bounds)
            elif isinstance(layer, nn.ReLU):
                # Compute the post-activation bounds for the ReLU activation
                post_bounds = compute_relu_bounds(pre_bounds)

        # Return the final bounds on the input layer
        return pre_bounds

def compute_linear_bounds(layer, post_bounds):
    # Implement the computation of the pre-activation bounds for the linear layer
    pass

def compute_relu_bounds(pre_bounds):
    # Implement the computation of the post-activation bounds for the ReLU activation
    pass

# Create a CROWN model and compute the backward bounds
model = CrownModel()
output_bounds = (torch.tensor([[0.0]]), torch.tensor([[1.0]]))
input_bounds = model.backward_bound_propagation(output_bounds)
print("Input bounds:", input_bounds)