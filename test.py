from customppo import CustomPPO
from customac import CustomAC

import torch as th
import argparse
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO

import pickle
import gymnasium as gym
import os
import numpy as np
from tqdm import tqdm, trange
import time

def main(args):

    vec_env = make_vec_env(args.env_name, n_envs=1)

    if args.modelpath == None:
        args.modelpath = "./saves/training/LunarLander-v2-0.1-True-1689218583-ts-2880000"
    
    model = PPO.load(args.modelpath)

    obs = vec_env.reset()
    dones = False
    rwd_list = list()
    iter = 1000
    pbar = tqdm(total=iter, desc='Processing', unit='iteration')

    step_eps = args.epsilon / args.steps
    clamp_min = th.tensor(obs - args.epsilon).to('cuda')
    clamp_max = th.tensor(obs + args.epsilon).to('cuda')

    if not os.path.exists(args.rewardpath):
        os.makedirs(args.rewardpath)
    
    for i in range(iter):
        total_reward = 0
        while not dones:

            # Critic Attack
            # =====================================================================
            if args.epsilon != 0:           
                noise = np.random.uniform(-step_eps, step_eps, size=obs.shape)
                states = th.tensor(obs + noise).to('cuda')
                with th.enable_grad():
                    for i in range(args.steps):
                        states = states.clone().detach().requires_grad_()
                        value = model.policy.predict_values(states)
                        value.backward()
                        update = states.grad.sign() * step_eps
                        states.data = th.min(th.max(states.data - update, clamp_min), clamp_max)
                    model.policy.mlp_extractor.value_net.zero_grad()
                obs = states.detach().cpu().numpy()
            # =====================================================================
            
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            total_reward += rewards
            if args.render:
                vec_env.render("human")
        pbar.set_postfix({"Episode Reward": total_reward})
        pbar.update(1)
        rwd_list.append(total_reward)
        dones = False
        
    print("\n")
    with open(args.rewardpath + '/reward.pkl', 'wb') as file:
        pickle.dump(rwd_list, file)
        print("File Saved")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Customized PPO')
    parser.add_argument('--bound', type=int, default=0,
	                    help='To use bounded value net or not')
    parser.add_argument('--epsilon', type=float, default=0.0,
	                    help='Amount of perturbation added to the system')
    parser.add_argument('--env-name', type=str, default="LunarLander-v2")
    parser.add_argument('--iter-num', type=int, default=50)
    parser.add_argument('--per-iter-step', type=int, default=80000)
    parser.add_argument('--use-wandb', type=int, default=0)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--rewardpath', type=str, default="./reward")


    args = parser.parse_args()
    print(args)
    main(args)
