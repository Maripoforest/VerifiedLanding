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
with open('reward_per_bou_12_12.pkl', 'rb') as file:
    reward_per_bou_12_12 = pickle.load(file)
with open('reward_per_bou_12_2.pkl', 'rb') as file:
    reward_per_bou_12_2 = pickle.load(file)
# with open('reward/reward_per_bou_1.pkl', 'rb') as file:
#     reward_per_bou_1 = pickle.load(file)
# with open('reward/reward_no_per.pkl', 'rb') as file:
#     reward_no_per = pickle.load(file)
# data = np.sort(data)
# plt.plot(data)

# sns.kdeplot(reward_per_nobou_2, label = 'no_bound_0.2')
# sns.kdeplot(reward_no_per, label = 'no_per')
# sns.kdeplot(reward_per_nobou, label = 'no_bound_0.1')
# sns.kdeplot(reward_per_bou_2, label = 'bound_0.2')
sns.kdeplot(reward_per_bou_12_2, label = 'bound0.12_per0.2')
sns.kdeplot(reward_per_bou_12_12, label = 'bound0.12_per0.12')
# sns.kdeplot(reward_per_bou_1, label = 'bound_0.1')

plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Density')
# plt.title('per_no_bound_0.2')
plt.show()