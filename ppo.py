# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:40:21

@author: Xiangmin
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
            nn.ReLU()
        )
        # Cost network
        self.cost_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_co), 
            nn.ReLU(),
            nn.Linear(last_layer_dim_co, last_layer_dim_co), 
            nn.ReLU(),
        )
        
        self.bounded_value_net = IBPModel(self.value_net)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        
        self.bounded_value_net.load_state_dict(self.value_net.state_dict())
        l, u =  self.bounded_value_net.compute_bounds(features)
        
        # return self.forward_actor(features), self.forward_critic(features), self.forward_cost(features)
        return self.forward_actor(features), l


    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
    
    # def forward_cost(self, features: th.Tensor) -> th.Tensor:
    #     return self.cost_net(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
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

    # env = make_vec_env("LunarLander-v2", n_envs=1)

    # model = PPO(ActorCriticPolicy, env, verbose=1,tensorboard_log="./tsb/ppo_LunarLander_tensorboard/")
    # model.learn(
    #     total_timesteps=500000,
    #     tb_log_name="first_run"
    #     )
    # model.save("ppo_LunarLander")
    # del model # remove to demonstrate saving and loading

    env = make_vec_env("LunarLander-v2", n_envs=1)
    model = PPO.load("ppo_LunarLander_bound")
    obs = env.reset()
    i = 0
    rewards = 0
    all_reward = []

    while i < 300:
        perturbations = np.random.uniform(-0.1, 0.1, obs.shape)
        perturbations[0][6:] = 0
        obs += perturbations
        tobs = th.tensor(obs)
        action, _states = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        rewards += reward[0]
        if(dones):
            i += 1
            print("Run:", i)
            print(rewards)
            all_reward.append(rewards)
            rewards = 0
        env.render()
    with open('reward_per_bou.pkl', 'wb') as f:
        pickle.dump(all_reward, f)
