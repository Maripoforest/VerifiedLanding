from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
import numpy as np

class CustomBoundedNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        last_layer_dim_co: int = 64,
        epsilon: float = 0.0,
        bound: bool = False
        ):
        super().__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.latent_dim_co = last_layer_dim_co

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), 
            nn.LeakyReLU(), 
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
            nn.LeakyReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), 
            nn.LeakyReLU(),
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf), 
            nn.LeakyReLU(),
            nn.Linear(last_layer_dim_vf, 1)
        )
        # Cost network
        self.cost_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_co), 
            nn.LeakyReLU(),
            nn.Linear(last_layer_dim_co, last_layer_dim_co), 
            nn.LeakyReLU(),
            nn.Linear(last_layer_dim_co, last_layer_dim_co), 
        )

        self.epsilon = epsilon
        self.forward_critic = self.forward_critic_unbounded
        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            print("Using CPU")
        for param in self.value_net.parameters():
            th.nn.init.zeros_(param)

    def build_forward(self, bound, epsilon) -> None:
        if bound == 1:
            print("Using bounded value net")
            self.forward_critic = self.forward_critic_bounded
        else:
            print("Using unbounded value net")
            self.forward_critic = self.forward_critic_unbounded
        self.epsilon = epsilon
    
    def forward_critic_unbounded(self, features: th.Tensor, is_lower: bool = True) -> th.Tensor:
        return self.value_net(features)
    
    def forward_critic_bounded(self, features: th.Tensor, is_lower: bool = True) -> th.Tensor:
        if is_lower:
            l, u = self.compute_bounds(features, self.value_net)
        else:
            l = self.value_net(features)
        return l
    
    def forward_actor(self, features : th.Tensor) -> th.Tensor:
        return self.policy_net(features)
    
    def forward(self, features : th.Tensor, is_lower: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features, is_lower=is_lower)

    def compute_bounds(self, features: th.Tensor, network: nn.Sequential) -> Tuple[th.Tensor, th.Tensor]:
        l = th.full_like(features, -self.epsilon).to(self.device)
        u = th.full_like(features, self.epsilon).to(self.device)
        l += features
        u += features
        for layer in network:
            if isinstance(layer, nn.Linear):
                l, u = self.interval_bound_propagation(layer, l, u)
            else:
                l = layer(l)
        return l, u
    
    def interval_bound_propagation(self, layer: nn.Linear, l: th.Tensor, u: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        W = layer.weight
        b = layer.bias
        l_out = th.matmul(l, W.clamp(min=0).t()) + th.matmul(u, W.clamp(max=0).t()) + b
        u_out = th.matmul(u, W.clamp(min=0).t()) + th.matmul(l, W.clamp(max=0).t()) + b
        return l_out, u_out
