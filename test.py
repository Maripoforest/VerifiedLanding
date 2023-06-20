from customppo import CustomPPO
import pickle
import torch as th
import argparse
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
    parser.add_argument('--test-num', type=int, default=300)
    parser.add_argument('--per-iter-step', type=int, default=80000)
    parser.add_argument('--use-wandb', type=int, default=0)
    parser.add_argument('--model-path', type=str, default="")
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--render', type=int, default=0)

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    name = make_name(args)
    args.name = name
    print(name)

    env = make_vec_env(args.env_name, n_envs=1)
    model = CustomPPO.load(args.model_path)
    model.check_status()

    obs = env.reset()
    pbar = tqdm(iterable=range(args.test_num))

    rewards = 0
    all_reward = list()
    i = 0

    while i < args.test_num:
        perturbations = np.random.choice([-1, 1], size=obs.shape) * args.epsilon
        obs += perturbations
        action, states = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        rewards += reward[0]
        if dones:
            i += 1
            print("Run: ", i)
            print("Episode Reward:", rewards)
            all_reward.append(rewards)
            rewards = 0
        if args.render == 1:
            env.render()
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(all_reward, f)