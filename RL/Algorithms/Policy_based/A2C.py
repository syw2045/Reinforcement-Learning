import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

HIDDEN_SIZE = 128
EPISODES = 100
BATCH_SIZE = 5 
ENTROPY_BETA = 0.01

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

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

        self.memory = []

    def get_action(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs[action], probs
    
    def store_transition(self, state, action_prob, reward, next_state, done, probs):
        """ 배치 업데이트를 위해 경험을 저장 """
        self.memory.append((state, action_prob, reward, next_state, done, probs))
    
    def update(self):
        """ 배치 단위 업데이트 수행 """
        if len(self.memory) < BATCH_SIZE:
            return

        states, action_probs, rewards, next_states, dones, probs_list = zip(*self.memory)
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        target_values = rewards + self.gamma * self.v(next_states).squeeze(1) * (1 - dones)
        values = self.v(states).squeeze(1)
        loss_v = nn.MSELoss()(values, target_values.detach())

        advantages = (target_values - values).detach()

        log_probs = torch.log(torch.stack(action_probs))
        loss_pi = -torch.mean(log_probs * advantages)


        entropy = -torch.mean(torch.stack([torch.sum(p * torch.log(p + 1e-5)) for p in probs_list]))
        loss_pi -= ENTROPY_BETA * entropy 


        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()

        self.memory = [] 


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
            action, prob, probs = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            agent.store_transition(state, prob, reward, next_state, done, probs)
            state = next_state
            total_reward += reward

            if done:
                agent.update()
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.1f}")

    env.close()
