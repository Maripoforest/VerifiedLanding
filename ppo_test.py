# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:49:49 2022

@author: Xiangmin
"""

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import pickle
import torch as th
from torch import nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ibp_relu import IBPModel

def discount(lst, gamma=0.99):
    n = len(lst)
    discounts = np.array([gamma**i for i in range(n)])
    longterm = np.zeros(n)
    for t in range(n):
        longterm[t] = lst[t] * discounts[n-t-1]
    return np.sum(longterm)


if __name__ == "__main__":
    
    eps = 10
    bound = 1
    per = True
    sec = False
    upper = False
    
    if eps != 0:
        epsn = str(eps)
        if bound:
            modelname = "trained/ppo_LunarLander_bound_per" + epsn
            rewardname = "reward/reward_per_bou_" + epsn
            speedname = "reward/speed_per_bou_" + epsn
        else:
            modelname = "trained/ppo_LunarLander_nobound_per" + epsn
            rewardname = "reward/reward_per_nobou_" + epsn
            speedname = "reward/speed_per_nobou_" + epsn
    else:
        modelname = "trained/ppo_LunarLander_nobound_noper"
        rewardname = "reward/reward_noper"
        speedname = "reward/speed_noper"
        
    if sec:
        modelname += "_v2"
        rewardname += "_v2"
        speedname += "_v2"
    if upper:
        modelname += "_upper"
        rewardname += "_upper"
        speedname += "_upper"


    env = make_vec_env("LunarLander-v2", n_envs=1)
    model = PPO.load(modelname)
    model.policy.mlp_extractor._training = False
    print(model.policy.mlp_extractor._training)
    obs = env.reset()
    i = 0
    rewards = 0
    all_reward = []
    vels = []
    vel = 0
    j = 0

    print("testing:", modelname)
    while i < 100:
        # if obs[0][6] != 1 and obs[0][7] != 1:
        #     j += 1
        #     vel += np.sqrt((obs[0][2])**2 + (obs[0][3])**2)
        
        perturbations = np.random.uniform(-eps/100.0, eps/100.0, obs.shape)
        # perturbations = np.random.choice([-1, 1], size=obs.shape) * eps / 100
        perturbations[0][6:] = 0
        obs += perturbations
        action, _states = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        
        rewards += reward[0]

        if(dones):
            i += 1
            # vel /= j
            print("Run:", i)
            print("Reward:", rewards)
            # print("Average Speed:", vel)
            all_reward.append(rewards)
            # vels.append(vel)
            rewards = 0
            # j = 0
            # vel = 0
        env.render()
    with open(rewardname + '.pkl', 'wb') as f:
        pickle.dump(all_reward, f)
    # with open(speedname + '.pkl', 'wb') as f:
    #     pickle.dump(vels, f)