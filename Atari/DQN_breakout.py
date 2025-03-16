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

# Hyperparameters
ACTION_SIZE = 4  # Noop / Fire / Right / Left

MAX_EPISODE = 20000  # Max episode
LOAD_EPISODE = 2000

LOAD_MODEL = True
TRAIN_MODE = False

BATCH_SIZE = 32
MEM_MAXLEN = 200000
MEM_MINLEN = 60000
START_EPISODE = 1
CAREER_HIGH = 0

discount_factor = 0.99
learning_rate = 0.0001

target_update_step = 10000

SAVE_INTERVAL = 250

epsilon_eval = 0.05
epsilon_init = 1.0 if TRAIN_MODE else epsilon_eval
epsilon_min = 0.02

explore_step = 1000000. if TRAIN_MODE else 0
epsilon_delta = (epsilon_init - epsilon_min)/explore_step if TRAIN_MODE else 0

# Model save and load path
date_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
save_path = f"./saved_models/Atari/BreakOut/DQN/{date_time}"
load_path = "./saved_models/Atari/BreakOut/DQN/250315234955"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN
class DQN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = torch.nn.Linear(7*7*64, 512)
        self.q = torch.nn.Linear(512, ACTION_SIZE)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.q(x)
        return x

# DQN_Agent
class DQNAgent:
    def __init__(self):
        self.network = DQN().to(DEVICE)
        self.target_network = DQN().to(DEVICE)
        self.update_target()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=MEM_MAXLEN)
        self.epsilon = epsilon_init
        self.writer = SummaryWriter(save_path)

        if LOAD_MODEL == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=DEVICE)
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

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


    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append([state[None, :], action, reward, next_state[None, :], done])

    def train_model(self):
        if len(self.memory) < MEM_MINLEN:
            return 0, 0

        state, action, reward, next_state, done = zip(*random.sample(self.memory, BATCH_SIZE))

        state = np.concatenate(state)
        next_state = np.concatenate(next_state)

        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.long, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float, device=DEVICE)

        q_values = self.network(state)
        q_values = q_values.gather(1, action.unsqueeze(1))

        future_rewards = self.target_network(next_state)

        updated_q_values = reward + discount_factor * future_rewards.max(1)[0] * (1 - done)

        loss = F.smooth_l1_loss(q_values.squeeze(1), updated_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        self.epsilon = max(epsilon_min, self.epsilon - epsilon_delta)

        return loss.item(), torch.max(q_values).item()

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def save_model(self, episode, reward):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, save_path + f"/ckpt_{episode}_{int(reward)}")

    def write_summray(self, score, loss, q_val, episode, epsilon):
        self.writer.add_scalar("run/score", score, episode)
        self.writer.add_scalar("model/loss", loss, episode)
        self.writer.add_scalar("model/q_val", q_val, episode)
        self.writer.add_scalar("model/epsilon", epsilon, episode)

    def preprocessing(self, state):
        frame = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        frame = frame[34:194, 0:160]
        frame = cv2.resize(frame, (84, 84))
        frame = frame / 255.0
        return frame

if __name__ == '__main__':
    env = gym.make('ALE/Breakout-v5', render_mode="human") # rgb_array
    agent = DQNAgent()

    losses, scores, episode, score = [], [], 0, 0
    avg_100_reward = deque(maxlen=100)
    total_step = 0

    for episode in range(START_EPISODE, MAX_EPISODE):
        dead = False
        done = False

        total_reward, total_loss, total_q_val, game_life = 0, 0, 0, 5

        state = env.reset()

        for _ in range(random.randint(1, 30)):
            state, _, _, _, _ = env.step(1) # state : (210, 160, 3)
        state = agent.preprocessing(state)
        state = np.stack((state,state,state,state))

        while not done:
            total_step += 1

            if dead:
                dead = False
                action = 1
            else: action = agent.get_action(state)

            next_state, reward, done, _ , info= env.step(action)

            next_state = agent.preprocessing(next_state)
            next_state = np.stack((next_state, state[0], state[1], state[2]))

            if game_life > info['lives']:
                dead = True
                game_life = info['lives']
            else: dead = done

            agent.append_sample(state, action, np.clip(reward, -1. , 1.), next_state, dead)
            state = next_state

            if TRAIN_MODE:
                loss, max_q_val = agent.train_model()
                if total_step % target_update_step == 0:
                    agent.update_target()
            else:
                loss, max_q_val = [0, 0]

            total_loss += loss
            total_q_val += max_q_val
            total_reward += reward

            if done:
                current_time_format = time.strftime("%H:%M:%S", time.localtime())
                avg_100_reward.append(total_reward)
                avg_reward = np.mean(avg_100_reward)
                CAREER_HIGH = max(CAREER_HIGH, avg_reward)
                env.reset()

                if episode % 10 == 0:
                    print(f"{episode} Episode / Time : {current_time_format} /  Total_Step: {total_step} / Score: {total_reward} / avg_reward: {avg_reward:.3f} / Loss: {total_loss:.2f} / Q_val: {total_q_val:.2f} / Epsilon: {agent.epsilon:.4f}, career_high: {CAREER_HIGH:.3f}")

                agent.write_summray(avg_reward, total_loss, total_q_val, episode, agent.epsilon)

                if TRAIN_MODE and episode % SAVE_INTERVAL == 0:
                    agent.save_model(episode, avg_reward)

    env.close()
