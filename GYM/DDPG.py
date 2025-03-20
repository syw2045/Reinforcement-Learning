import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import copy
import datetime

from torch.utils.tensorboard import SummaryWriter
from collections import deque

STATE_DIM = 2
ACTION_DIM = 1

MAX_EPISODE = 1000
MAX_STEP = 999

ACTOR_lr = 0.0001
CRITIC_lr = 0.001
GAMMA = 0.99
TAU = 0.001
MU = 0.0
THETA = 0.15
SIGMA = 0.2

MEM_MAXLEN = 100000
MEM_MINLEN = 1000

BATCH_SIZE = 32

TEST_MODE = False
TRAIN_MODE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# date_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
# save_path = f"./saved_models/MountainCar/DDPG/{date_time}"
# load_path = f"./saved_models/MountainCar/DDPG/"

class OU_noise:
    def __init__(self, size=ACTION_DIM, mu=MU, theta=THETA, sigma=SIGMA):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(STATE_DIM, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, ACTION_DIM)
        self.tanh = torch.nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.tanh(self.fc3(x))

class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(STATE_DIM + ACTION_DIM, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.q = torch.nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.q(x)

class ReplayMemory:
    def __init__(self):
        self.memory = deque(maxlen=MEM_MAXLEN)
    
    def append_sample(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def size(self):
        return len(self.memory)

class DDPGAgent:
    def __init__(self):
        self.actor = Actor().to(DEVICE)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_lr)

        self.critic = Critic().to(DEVICE)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_lr)

        self.OU = OU_noise()
        self.memory = ReplayMemory()

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        noise_sample = self.OU.sample()
        return np.clip(action + noise_sample, -1.0, 1.0)

    def append_sample(self, transition):
        self.memory.append(transition)

    def train_model(self):
        if self.memory.size() < MEM_MINLEN:
            return 0, 0

        state, action, reward, next_state, done = zip(*self.memory.sample(BATCH_SIZE))

        state = torch.tensor(np.array(state), dtype=torch.float32)
        action = torch.tensor(np.array(action), dtype=torch.float32)
        reward = torch.tensor(np.array(reward), dtype=torch.float32).unsqueeze(1)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        done = torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(1)
        
        target = reward + GAMMA * self.target_critic(next_state, self.target_actor(next_state)) * (1-done)
        
        # Critic Loss
        critic_loss = torch.nn.MSELoss()(self.critic(state, action), target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.critic_optimizer.step()

        # Actor Loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.actor_optimizer.step()

        return actor_loss, critic_loss

    def soft_update_target(self):
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    agent = DDPGAgent()
    total_step = 0

    for episode in range(1, MAX_EPISODE):
        state, _ = env.reset()
        agent.OU.reset()
        actor_loss, critic_loss, total_reward = 0, 0, 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated

            agent.memory.append_sample((state, action, reward, next_state, done))
            total_reward += reward
            total_step += 1
            state = next_state
            if TRAIN_MODE:
                actor_loss, critic_loss = agent.train_model()
                agent.soft_update_target()
            


        print(f"Episode {episode} | Total Steps {total_step} | Avg Reward: {total_reward:.2f} | Actor_Loss:{actor_loss:.4f} | Critic_Loss:{critic_loss:.4f}")

    env.close()
