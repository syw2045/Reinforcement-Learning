import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim



Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 500

class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output

BATCH_SIZE = 32
CAPACITY = 10000

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)

        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)
        self.target_q_network = Net(n_in, n_mid, n_out)
        print(self.main_q_network)

        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=0.0001)
        
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()
        self.expected_q_function = self.get_expected_q_function()
        self.update_main_q_network()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])
        return action
    
    def make_minibatch(self):
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        return batch, state_batch, action_batch, reward_batch, non_final_next_states
    
    def get_expected_q_function(self):
        self.main_q_network.eval()
        self.target_q_network.eval()

        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)
        
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)
        action_max = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        action_max[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]
        action_max_final_next_states = action_max[non_final_mask].view(-1, 1)

        next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states).gather(1, action_max_final_next_states).detach().squeeze()
        expected_q_function = self.reward_batch + GAMMA * next_state_values.detach()
        return expected_q_function
    
    def update_main_q_network(self):
        self.main_q_network.train()
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_q_function.unsqueeze(1).requires_grad_())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        return self.brain.decide_action(state, episode)
    
    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()

class Environment:
    def __init__(self):
        self.env = gym.make("CartPole-v1", render_mode="human")
        self.env.reset()
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions)

    def run(self):
        for episode in range(NUM_EPISODES):
            observation, _ = self.env.reset()
            state = torch.from_numpy(observation).float().unsqueeze(0)
            for step in range(MAX_STEPS):
                action = self.agent.get_action(state, episode)
                observation_next, _, done, _, _ = self.env.step(action.item())
                state_next = None if done else torch.from_numpy(observation_next).float().unsqueeze(0)
                self.agent.memorize(state, action, state_next, torch.FloatTensor([1.0 if not done else -1.0]))
                self.agent.update_q_function()
                state = state_next
                if done:
                    break
            if episode % 2 == 0:
                self.agent.update_target_q_function()

if __name__ == "__main__":
    cartpole_env = Environment()
    cartpole_env.run()
