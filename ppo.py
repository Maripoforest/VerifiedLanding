# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:49:49 2022

@author: 29134
"""

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import pickle
import torch as th
from torch import nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env

from ibp_relu import IBPModel

# def discount(lst, gamma=0.99):
#     n = len(lst)
#     discounts = np.array([gamma**i for i in range(n)])
#     longterm = np.zeros(n)
#     for t in range(n):
#         longterm[t] = lst[t] * discounts[n-t-1]
#     return np.sum(longterm)


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        last_layer_dim_co: int = 64,
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
            nn.ReLU(), 
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
            nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), 
            nn.ReLU(),
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf), 
            nn.ReLU(),
            nn.Linear(last_layer_dim_vf, 1)
        )
        # Cost network
        self.cost_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_co), 
            nn.ReLU(),
            nn.Linear(last_layer_dim_co, last_layer_dim_co), 
            nn.ReLU(),
        )
        self.epsilon = 0.16

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        
        # return self.forward_actor(features), self.forward_critic(features), self.forward_cost(features)
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        
        perturbations = (th.rand_like(features) - 0.5) * 2 * self.epsilon
        perturbations[:][6:] = 0
        perturbations += features
        return self.policy_net(perturbations)
    
        # return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        
        # perturbations = (th.rand_like(features) - 0.5) * 2 * self.epsilon
        # perturbations[:][6:] = 0
        # perturbations += features
        # return self.value_net(perturbations)
    
        l, u =  self.compute_bounds(features)
        return l

        # return self.value_net(features)
        
    def forward_cost(self, features: th.Tensor) -> th.Tensor:
        return self.cost_net(features)

    def compute_bounds(self, features):
        # if(len(features) == 1):
        #     x_bounds = features[0]
        #     l = th.full_like(x_bounds, -self.epsilon)
        #     l[6:] = 0
        #     u = th.full_like(x_bounds, self.epsilon)
        #     u[6:] = 0
        #     l += x_bounds
        #     u += x_bounds
        #     for layer in self.value_net:
        #         if isinstance(layer, nn.Linear):
        #             l, u = self.interval_bound_propagation(layer, l, u)
        #         elif isinstance(layer, nn.ReLU):
        #             # l, u = self.relu_relaxaion_approximation(l, u)
        #             l = self.relu(l)
        #             u = self.relu(u)
        #     return l, u
        # else:
        ls = []
        us = []
        for x_bounds in features:
            l = th.full_like(x_bounds, -self.epsilon)
            l[6:] = 0
            u = th.full_like(x_bounds, self.epsilon)
            u[6:] = 0
            l += x_bounds
            u += x_bounds
            for layer in self.value_net:
                if isinstance(layer, nn.Linear):
                    l, u = self.interval_bound_propagation(layer, l, u)
                elif isinstance(layer, nn.ReLU):
                    # l, u = self.relu_relaxaion_approximation(l, u)
                    l = self.relu(l)
                    u = self.relu(u)
            ls.append(l)
            us.append(u)
        ls = th.stack(ls, dim=0)
        us = th.stack(us, dim=0)

        return ls, us

    def interval_bound_propagation(self, layer, l, u):
        W, b = layer.weight, layer.bias
        l_out = th.matmul(W.clamp(min=0), l) + th.matmul(W.clamp(max=0), u) + b
        u_out = th.matmul(W.clamp(min=0), u) + th.matmul(W.clamp(max=0), l) + b
        return l_out, u_out
    
    def relu_relaxaion_approximation(self, l, u, alpha=0.5):
        slope = (u - l) / (alpha * (u - l) + (1 - alpha) * (l + u))
        bias = l - slope * l
        l_out = slope * l + bias
        u_out = slope * u + bias
        return l_out, u_out

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)



if __name__ == "__main__":

    env = make_vec_env("LunarLander-v2", n_envs=1)
    # model = PPO(ActorCriticPolicy, env, verbose=1,tensorboard_log="./tsb/ppo_LunarLander_tensorboard/")
    model = PPO(CustomActorCriticPolicy, env, verbose=1,tensorboard_log="./tsb/ppo_LunarLander_tensorboard/")


    for i in range(100):
        model.learn(
            total_timesteps=30000,
            tb_log_name="nobound",
            reset_num_timesteps=False
            )
        model.save("ppo_LunarLander_nobound_per23_timestep"+str(i*30000))
    del model # remove to demonstrate saving and loading

    # env = make_vec_env("LunarLander-v2", n_envs=1)
    # model = PPO.load("trained/ppo_LunarLander_noper")
    # obs = env.reset()
    # i = 0
    # rewards = 0
    # all_reward = []

    # while i < 300:
    #     # perturbations = np.random.uniform(-0.1, 0.1, obs.shape)
    #     # # perturbations[0][6:] = 0
    #     # obs += perturbations
    #     action, _states = model.predict(obs)
    #     obs, reward, dones, info = env.step(action)
    #     rewards += reward[0]
    #     if(dones):
    #         i += 1
    #         print("Run:", i)
    #         print(rewards)
    #         all_reward.append(rewards)
    #         rewards = 0
    #     env.render()
    # with open('reward_no_per.pkl', 'wb') as f:
    #     pickle.dump(all_reward, f)