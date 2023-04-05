import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# with open('reward/reward_per_nobou_2.pkl', 'rb') as file:
#     reward_per_nobou_2 = pickle.load(file)
# with open('reward/reward_per_nobou_1.pkl', 'rb') as file:
#     reward_per_nobou = pickle.load(file)
# with open('reward/reward_per_bou_2_v2.pkl', 'rb') as file:
#     reward_per_bou_2 = pickle.load(file)
# with open('reward/reward_per_bou_12_12.pkl', 'rb') as file:
#     reward_per_bou_12_12 = pickle.load(file)
# with open('reward/reward_per_bou_12_2.pkl', 'rb') as file:
#     reward_per_bou_12_2 = pickle.load(file)
# with open('reward/reward_per_bou_1.pkl', 'rb') as file:
#     reward_per_bou_1 = pickle.load(file)
# with open('reward/reward_no_per.pkl', 'rb') as file:
#     reward_no_per = pickle.load(file)
# with open('reward/reward_per_bou_16.pkl', 'rb') as file:
#     reward_per_bou_16 = pickle.load(file)
# with open('reward/reward_per_nobou_16.pkl', 'rb') as file:
#     reward_per_nobou_16 = pickle.load(file)
# with open('reward/reward_per_bou_14.pkl', 'rb') as file:
#     reward_per_bou_14 = pickle.load(file)
# with open('reward/reward_per_nobou_14.pkl', 'rb') as file:
#     reward_per_nobou_14 = pickle.load(file)
with open('reward/reward_per_bou_18.pkl', 'rb') as file:
    reward_per_bou_18 = pickle.load(file)
# data = np.sort(data)
# plt.plot(data)

sns.kdeplot(reward_per_bou_18, label = 'bound_0.18')
# sns.kdeplot(reward_per_bou_14, label = 'bound_0.14')
# sns.kdeplot(reward_per_nobou_14, label = 'no_bound_0.14', color='r', linestyle = "--")
# sns.kdeplot(reward_per_bou_16, label = 'bound_0.16')
# sns.kdeplot(reward_per_nobou_16, label = 'no_bound_0.16', color='r', linestyle = "--")
# sns.kdeplot(reward_per_nobou_2, label = 'no_bound_0.2')
# sns.kdeplot(reward_no_per, label = 'no_per')
# sns.kdeplot(reward_per_nobou, label = 'no_bound_0.1')
# sns.kdeplot(reward_per_bou_2, label = 'bound_0.2')
# sns.kdeplot(reward_per_bou_12_2, label = 'bound0.12_per0.2', color='r', linestyle = "--")
# sns.kdeplot(reward_per_bou_12_12, label = 'bound0.12_per0.12')
# sns.kdeplot(reward_per_bou_1, label = 'bound_0.1')

plt.legend()
plt.xlabel('Reward')
plt.ylabel('Density')
# plt.title('per_no_bound_0.2')
plt.show()