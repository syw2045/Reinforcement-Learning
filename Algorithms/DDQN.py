import numpy as np
import matplotlib.pyplot as plt
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
from collections import namedtuple
import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn
from torch import optim

def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0),
               dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
                                   interval=50)

    anim.save('movie_cartpole_DDQN.mp4')
    display(display_animation(anim, default_mode='loop'))

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 500

# ReplayMemory
class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, aciton, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    
    def __len__(self):
        return len(self.memory)
    
# 신경망
class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self,x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output

BATCH_SIZE = 32
CAPACITY = 10000

class Brain:
    def __init__(self):
        pass