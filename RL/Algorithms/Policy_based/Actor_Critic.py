import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

HIDDEN_SIZE = 128
EPISODES = 100

class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)
    
class ValueNet(nn.Module):
    def __init__(self, state_size):
        super(ValueNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, state):
        return self.network(state) 
    
class Agent:
    def __init__(self, state_size, action_size):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.state_size = state_size
        self.action_size = action_size

        self.pi = PolicyNet(self.state_size, self.action_size)
        self.v = ValueNet(self.state_size)

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr = self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        state = torch.tensor(state[np.newaxis, :])
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs[action]
    
    def update(self, state, action_prob, reward, next_state, done):
        state = torch.tensor(state[np.newaxis, :])
        next_state = torch.tensor(next_state[np.newaxis, :])

        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.detach()
        v = self.v(state)
        loss_fn = nn.MSELoss()
        loss_v = loss_fn(v, target)

        delta = target - v
        loss_pi = -torch.log(action_prob) * delta.item()

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)

    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            agent.update(state, prob, reward, next_state, done)

            state = next_state
            total_reward += reward

        if episode % 10 == 0:
            print("episode :{}, total reward : {:.1f}".format(episode, total_reward))

    env.close()