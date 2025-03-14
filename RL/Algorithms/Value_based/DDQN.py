from collections import deque
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HIDDEN_SIZE = 128
NUM_EPISODES = 200
SYNC_INTERVAL = 20


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in data]))
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.compat.long))
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))
        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.l1 = nn.Linear(state_size, HIDDEN_SIZE)
        self.l2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.value = nn.Linear(HIDDEN_SIZE, action_size)

    def forward(self, state):
        l1 = F.relu(self.l1(state))
        l2 = F.relu(self.l2(l1))
        value = self.value(l2)
        return value


class DDQN_Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        
        self.epsilon_decay = 0.999
        self.epsilon = 1.0
        self.epsilon_end = 0.1
        
        self.buffer_size = 10000
        self.batch_size = 32
        self.state_size = 4
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.state_size, self.action_size)
        self.qnet_target = QNet(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :])
            qs = self.qnet(state)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(len(action)), action]

        ### DDQN ###
        next_qs = self.qnet(next_state)
        next_actions = next_qs.argmax(1)

        next_qs_target = self.qnet_target(next_state)
        next_q = next_qs_target[np.arange(len(action)), next_actions]
        
        #######

        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())


class DDQN_Env:
    def __init__(self):
        self.env = gym.make("CartPole-v1", render_mode="human")
        self.env.reset()
        self.reward_history = []

    def run(self):
        for episode in range(NUM_EPISODES):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated | truncated

                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
            if episode % SYNC_INTERVAL == 0:
                agent.sync_qnet()

            self.reward_history.append(total_reward)
            if episode % 10 == 0:
                print("episode :{}, total reward : {}".format(episode, total_reward))

        self.env.close()
        

if __name__ == "__main__":
    env = DDQN_Env()
    agent = DDQN_Agent()
    env.run()