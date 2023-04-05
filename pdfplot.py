import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def mykdeplot(reward, label="reward", color='r', linestyle='-'):
    i = 0
    landed = 0
    success = 0
    for item in reward:
        i += 1
        if item > 100:
            landed += 1
        if item > 240:
            success += 1
    landed /= i
    success /= i
    label += ", lr: " + str(landed) + ", sr: " + str(success)
    sns.kdeplot(reward, label=label, color=color, linestyle=linestyle)



with open('reward/reward_per_nobou_20.pkl', 'rb') as file:
    reward_per_nobou_2 = pickle.load(file)
    
with open('reward/reward_per_bou_12_12.pkl', 'rb') as file:
    reward_per_bou_12_12 = pickle.load(file)
with open('reward/reward_per_bou_12_2.pkl', 'rb') as file:
    reward_per_bou_12_2 = pickle.load(file)

with open('reward/reward_per_bou_10.pkl', 'rb') as file:
    reward_per_bou_10 = pickle.load(file)
with open('reward/reward_per_nobou_10.pkl', 'rb') as file:
    reward_per_nobou_10 = pickle.load(file)

with open('reward/reward_no_per.pkl', 'rb') as file:
    reward_no_per = pickle.load(file)

with open('reward/reward_per_bou_16.pkl', 'rb') as file:
    reward_per_bou_16 = pickle.load(file)
with open('reward/reward_per_nobou_16.pkl', 'rb') as file:
    reward_per_nobou_16 = pickle.load(file)

with open('reward/reward_per_bou_14.pkl', 'rb') as file:
    reward_per_bou_14 = pickle.load(file)
with open('reward/reward_per_nobou_14.pkl', 'rb') as file:
    reward_per_nobou_14 = pickle.load(file)

with open('reward/reward_per_nobou_18.pkl', 'rb') as file:
    reward_per_nobou_18 = pickle.load(file)
with open('reward/reward_per_bou_18.pkl', 'rb') as file:
    reward_per_bou_18 = pickle.load(file)

with open('reward/reward_per_nobou_23.pkl', 'rb') as file:
    reward_per_nobou_23 = pickle.load(file)
with open('reward/reward_per_bou_23.pkl', 'rb') as file:
    reward_per_bou_23 = pickle.load(file)
# data = np.sort(data)
# plt.plot(data)


# mykdeplot(reward_per_bou_10, label = 'bound_0.10', color='purple', linestyle = "-")
# mykdeplot(reward_per_bou_14, label = 'bound_0.14', color='blue', linestyle = "--")
# mykdeplot(reward_per_bou_16, label = 'bound_0.16', color='orange', linestyle = ":")
mykdeplot(reward_per_bou_18, label = 'bound_0.18', color='r', linestyle = "-.")
mykdeplot(reward_per_bou_23, label = 'bound_0.23', color='blue', linestyle = "-")

# mykdeplot(reward_no_per, label = 'no_bound_no_per', color='black', linestyle = ":")
# mykdeplot(reward_per_nobou_10, label = 'no_bound_0.12', color='orange', linestyle = ":")
# mykdeplot(reward_per_nobou_14, label = 'no_bound_0.14', color='r', linestyle = "-.")
# mykdeplot(reward_per_nobou_16, label = 'no_bound_0.16', color='blue', linestyle = "--")
mykdeplot(reward_per_nobou_18, label = 'no_bound_0.18', color='purple', linestyle = "-")
mykdeplot(reward_per_nobou_23, label = 'no_bound_0.23', color='green', linestyle = "-")

# mykdeplot(reward_per_bou_12_2, label = 'bound0.12_per0.2', color='r', linestyle = "--")
# mykdeplot(reward_per_bou_12_12, label = 'bound0.12_per0.12', color='purple', linestyle = "-")

# sns.kdeplot(reward_per_bou_16, label = 'bound_0.16')
# mykdeplot(reward_per_nobou_16, label = 'no_bound_0.16', color='blue', linestyle = "--")
# sns.kdeplot(reward_per_nobou_2, label = 'no_bound_0.2')
# sns.kdeplot(reward_no_per, label = 'no_per')
# sns.kdeplot(reward_per_nobou, label = 'no_bound_0.1')
# sns.kdeplot(reward_per_bou_2, label = 'bound_0.2')
# mykdeplot(reward_per_bou_12_2, label = 'bound0.12_per0.2', color='r', linestyle = "--")
# mykdeplot(reward_per_bou_12_12, label = 'bound0.12_per0.12', color='purple', linestyle = "-")
# sns.kdeplot(reward_per_bou_1, label = 'bound_0.1')

plt.legend()
plt.xlabel('Reward (lr = Landing Rate, sr = Success Rate)')
plt.ylabel('Density')
# plt.title('per_no_bound_0.2')
plt.show()