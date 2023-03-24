import torch.nn as nn
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np

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
        self.initialize_parameters()
        
    def forward(self, x):
        return self.layers(x)
    
    def initialize_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0.1)
                nn.init.constant_(layer.bias, 0.1)
    
model = MyModel()
my_input = torch.tensor([0.5, 0.5])
# Wrap the model with auto_LiRPA.
model = BoundedModule(model, my_input)
# Define perturbation. Here we add Linf perturbation to input data.
ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
# Make the input a BoundedTensor with the pre-defined perturbation.
my_input = BoundedTensor(my_input, ptb)
# Regular forward propagation using BoundedTensor works as usual.
prediction = model(my_input)
# Compute LiRPA bounds using the backward mode bound propagation (CROWN).
lb, ub = model.compute_bounds(x=(my_input,), method="ibp")
print(lb, ub)