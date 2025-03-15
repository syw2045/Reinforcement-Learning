import gym
import numpy as np
import datetime
import torch
import torch.nn.functional as F
import copy
import random
import cv2

import time

from collections import deque
from torch.utils.tensorboard import SummaryWriter

ACTION_SIZE = 6
START_EPISODE = 1
MAX_EPISODE = 900
MAX_STEP = 100000
BATCH_SIZE = 64

SAVE_MODEL_INTERVAL = 10

LOAD_MODEL = True
TRAIN_MODE = False

mem_maxlen = 50000
mem_minlen = 40000
discount_factor = 0.99
learning_rate = 0.00025

target_update_step = 3000 # update 주기

PRINT_INTERVAL = 10
SAVE_INTERVAL = 10

epsilon_eval = 0.05
epsilon_init = 1.0 if TRAIN_MODE else epsilon_eval
epsilon_min = 0.01

explore_step = 50000 if TRAIN_MODE else 0
eplsilon_delta = (epsilon_init - epsilon_min)/explore_step if TRAIN_MODE else 0

# Model save and load path
date_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
save_path = f"./saved_models/Atari/Pong/DDQN/{date_time}"
load_path = "./saved_models/Atari/Pong/DDQN/250307201923"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDQN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DDQN, self).__init__(**kwargs)
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(64)

        self.fc1_adv = torch.nn.Linear(in_features= 4*6*64, out_features=512)
        self.relu_adv = torch.nn.LeakyReLU()
        self.fc2_adv = torch.nn.Linear(in_features=512, out_features=ACTION_SIZE)
        
        self.fc1_val = torch.nn.Linear(in_features= 4*6*64, out_features=512)
        self.relu_val = torch.nn.LeakyReLU()
        self.fc2_val = torch.nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        Ax = self.relu_adv(self.fc1_adv(x))
        Ax = self.fc2_adv(Ax)

        Vx = self.relu_val(self.fc1_val(x))
        Vx = self.fc2_val(Vx)
        
        Q = Vx + (Ax - Ax.mean())
        return Q


    
class DDQNAgent:
    def __init__(self, env):
        self.network = DDQN().to(DEVICE) # Network 생성
        self.target_network = copy.deepcopy(self.network) # Target Network 생성
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.epsilon = epsilon_init
        self.memory = deque(maxlen=mem_maxlen)
        self.writer = SummaryWriter(save_path)

        if LOAD_MODEL == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=DEVICE)
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def process_observation(self, obs):
        frame = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        frame = frame[20:210, 0:160]
        frame = cv2.resize(frame, (64, 80))
        frame = frame.reshape(64, 80) / 255
        return frame


    def get_action(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else epsilon_eval

        if epsilon > random.random():
            action = random.randrange(ACTION_SIZE)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
                q_values = self.network.forward(state)
                action = torch.argmax(q_values).item() 
        return action
    

    def train_model(self):
        if len(agent.memory) < mem_minlen:
            loss, max_q = [0, 0]
            return loss, max_q
        state, action, reward, next_state, done = zip(*random.sample(self.memory, BATCH_SIZE))

        state = np.concatenate(state)
        next_state = np.concatenate(next_state)

        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.long, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float, device=DEVICE)

        state_q_values = self.network(state)
        next_states_q_values = self.network(next_state)
        next_states_target_q_values = self.target_network(next_state)

        selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + discount_factor * next_states_target_q_value * (1 - done)

        loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(epsilon_min, self.epsilon - eplsilon_delta)
        return loss, torch.max(state_q_values).item()
            
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append([state[None, :], action, reward, next_state[None, :], done])

    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, save_path+'/ckpt')

    def write_summray(self, last_100_ep_reward, total_max_q_val, episode):
        self.writer.add_scalar("run/avg_score", last_100_ep_reward, episode)
        self.writer.add_scalar("model/qvalue", total_max_q_val, episode)


if __name__ == '__main__':
    env = gym.make("ALE/Pong-v5", render_mode="human")
    agent = DDQNAgent(env)

    startTime = time.time()  # Keep time
    total_loss, total_reward, total_step = [], [], 1
    last_100_ep_reward = deque(maxlen=100)

    for episode in range(START_EPISODE, MAX_EPISODE):
        obs, _ = env.reset() # (210, 160, 3) -> preprocessing (64, 80)
        obs = agent.process_observation(obs) # (64, 80)
        state = np.stack((obs, obs, obs, obs)) # (4, 64, 80)

        total_max_q_val = 0  # Total max q vals
        total_reward = 0  # Total reward for each episode
        total_loss = 0  # Total loss for each episode

        for step in range(MAX_STEP):
            total_step += 1
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = agent.process_observation(next_state)  # Process image
            next_state = np.stack((next_state, state[0], state[1], state[2]))
            agent.append_sample(state, action, reward, next_state, done)
            state = next_state
            if TRAIN_MODE:
                # Training
                loss, max_q_val = agent.train_model()
                if total_step % target_update_step == 0:
                    agent.update_target()
            else:
                loss, max_q_val = [0, 0]
            
            total_loss += loss
            total_max_q_val += max_q_val
            total_reward += reward
            
            if done:
                ###
                current_time_format = time.strftime("%H:%M:%S", time.localtime())
                last_100_ep_reward.append(total_reward)
                ###
                env.reset()

                print(f"{episode} Episode / Time : {current_time_format} /  Total_Step: {total_step} / Score: {total_reward} / avg_reward: {np.mean(last_100_ep_reward):.3f} / Loss: {total_loss:.2f} / Q_val: {total_max_q_val:.2f} / Epsilon: {agent.epsilon:.4f}")
                agent.write_summray(np.mean(last_100_ep_reward), total_max_q_val, episode)
                    
                if TRAIN_MODE and episode % SAVE_INTERVAL == 0:
                    agent.save_model()
                break

    env.close()
