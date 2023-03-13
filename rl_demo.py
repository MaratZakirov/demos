import gym
import numpy as np
import torch
from torch import nn

class navigator(nn.Module):
   def __init__(self, in_size=8, out_size=4):
       super().__init__()
       self.lr1 = nn.Linear(in_size, 64)
       self.lr2 = nn.Linear(64, 64)
       self.lr3 = nn.Linear(64, out_size)
       self.r = nn.ReLU()

   def forward(self, x):
       x = self.r(self.lr1(x))
       x = self.r(self.lr2(x))
       x = self.r(self.lr3(x))
       return x

env = gym.make("LunarLander-v2", render_mode="human")
obs_prev, info = env.reset(seed=42)

model = navigator()

for i in range(1000):
    action = env.action_space.sample()

    # this is where you would insert your policy
    obs_cur, reward, terminated, truncated, info = env.step(action)

    # if we became closer than give slightly positive feedback
    # otherwise give negative feedback
    action = model(torch.tensor(obs_cur))
    dist_prev = np.linalg.norm(obs_prev)
    dist_cur = np.linalg.norm(obs_cur)
    loss = nn.MSELoss(dist_cur - dist_prev, action)

    obs_prev = obs_cur

    if terminated or truncated:
        obs_prev, info = env.reset()
        break

env.close()