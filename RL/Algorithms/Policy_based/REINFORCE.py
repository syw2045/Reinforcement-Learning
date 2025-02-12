import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

HIDDEN_SIZE = 128
NUM_EPISODES = 1000

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.l1 = nn.Linear(state_size, HIDDEN_SIZE)
        self.l2 = nn.Linear(HIDDEN_SIZE, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class REINFORCE_Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.action_size = 2
        self.state_size = 4

        self.memory = []
        self.pi = Policy(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
            loss += - torch.log(prob) * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []

class REINFORCE_Env:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.reward_history = []

    def run(self):
        for episode in range(NUM_EPISODES):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action, prob = agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated | truncated

                agent.add(reward, prob)
                state = next_state
                total_reward += reward

            agent.update()

            self.reward_history.append(total_reward)
            if episode % 10 == 0:
                print("episode :{}, total reward : {:.1f}".format(episode, total_reward))

        self.env.close()

if __name__ == "__main__":
    env = REINFORCE_Env()
    agent = REINFORCE_Agent()
    env.run()