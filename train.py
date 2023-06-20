from customppo import CustomPPO
from customac import CustomAC

import torch as th
import argparse

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm
import numpy as np

def make_name(args):
    bound = True if args.bound==1 else False
    return str(args.env_name) + '-' + str(args.epsilon) + '-' +str(bound)

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


    args = parser.parse_args()
    print(args)

    name = make_name(args)
    args.name = name
    print(name)

    env = make_vec_env(args.env_name, n_envs=1)
    model = CustomPPO(CustomAC, env, verbose=1, tensorboard_log="./tsb/", batch_size=512)
    # model = PPO(ActorCriticPolicy, env, verbose=1, tensorboard_log="./tsb/")
    model.init_bounds(args)
    model.check_status()

    env.reset()
    pbar = tqdm(iterable=range(args.iter_num))
    for i in pbar:
        model.learn(
            total_timesteps=args.per_iter_step,
            tb_log_name=name,
            reset_num_timesteps=False        
            )
        model.save("./saves/training/" + name + "-ts-" + str(i*args.per_iter_step))
    del model
