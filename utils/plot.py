import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns


def pdfplot(data, legend: str = None):
    hist, bins = np.histogram(data, bins='auto', density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, hist, label=legend)

with open('./reward/reward_deterministic.pkl', 'rb') as file:
    d = pickle.load(file)
with open('./reward/reward_stochastic.pkl', 'rb') as file:
    s = pickle.load(file)

pdfplot(d, legend=f"deterministic mean={np.mean(d):.2f}")
pdfplot(s, legend=f"stochastic mean={np.mean(s):.2f}")

plt.legend()
plt.xlabel('Rewards')
plt.ylabel('Probability Density')
plt.title('PPO on LunarLander-v2')
plt.show()