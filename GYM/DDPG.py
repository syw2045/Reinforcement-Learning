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
discount_factor = 0.99
tau = 0.001
mu = 0.0
theta = 0.15
sigma = 0.2

MEM_MAXLEN = 100000
MEM_MINLEN = 10000

BATCH_SIZE = 64
SAVE_INTERVAL = 100

TEST_MODE = False
TRAIN_MODE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# date_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
# save_path = f"./saved_models/MountainCar/DDPG/{date_time}"
# load_path = f"./saved_models/MountainCar/DDPG/"

class OU_noise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X = np.ones((1, ACTION_DIM), dtype=np.float32) * mu

    def sample(self):
        dx = theta * (mu - self.X) + sigma * np.random.randn(len(self.X))
        self.X += dx
        return self.X
    
class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(STATE_DIM, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.mu = torch.nn.Linear(128, ACTION_DIM)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.nn.Tanh(self.mu(x))

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

    
class DDPGAgent:
    def __init__(self):
        self.actor = Actor().to(DEVICE)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_lr)
        self.critic = Critic().to(DEVICE)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_lr)
        self.OU = OU_noise()

        self.memory = deque(maxlen=MEM_MAXLEN)
        # self.writer = SummaryWriter(save_path)

        # 타겟 네트워크 초기화
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # if TEST_MODE == True:
        #     print(f"... Load Model from {load_path}/ckpt ...")
        #     checkpoint = torch.load(load_path+'/ckpt', map_location=DEVICE)
        #     self.actor.load_state_dict(checkpoint["actor"])
        #     self.target_actor.load_state_dict(checkpoint["actor"])
        #     self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        #     self.critic.load_state_dict(checkpoint["critic"])
        #     self.target_critic.load_state_dict(checkpoint["critic"])
        #     self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def get_action(self, state):
        action = self.actor(torch.tensor(state, dtype=torch.float)).item()
        noise = self.OU.sample().item()
        return np.clip(action + noise, -1.0, 1.0)  # 액션 범위 제한

    def append_sample(self, transition):
        self.memory.append(transition)

    def train_model(self):
        if len(self.memory) < MEM_MINLEN:
            return 0, 0

        mini_batch = random.sample(self.memory, BATCH_SIZE)
        state, action, reward, next_state, done = [torch.tensor(np.array(x), dtype=torch.float) for x in zip(*mini_batch)]
        
        target = reward + discount_factor * self.target_critic(next_state, self.target_actor(next_state)) * (1-done)
        
        # Critic Loss
        critic_loss = F.mse_loss(self.critic(state, action), target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss, critic_loss

    def soft_update_target(self):
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    # def save_model(self):
    #     print(f"... Save Model to {save_path}/ckpt ...")
    #     torch.save({
    #         "actor" : self.actor.state_dict(),
    #         "actor_optimizer" : self.actor_optimizer.state_dict(),
    #         "critic" : self.critic.state_dict(),
    #         "critic_optimizer" : self.critic_optimizer.state_dict(),
    #     }, save_path+'/ckpt')

    def write_summray(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)

if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    agent = DDPGAgent()

    avg_10_reward = deque(maxlen=10)
    total_step = 0

    for episode in range(1, MAX_EPISODE):
        state, _ = env.reset()
        agent.OU.reset()
        actor_loss, critic_loss, total_reward = 0, 0, 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step([action])
            done = terminated | truncated
            agent.append_sample((state, [action], [reward], next_state, [done]))
            total_reward += reward
            total_step += 1
            state = next_state

        if TRAIN_MODE:
            actor_loss, critic_loss = agent.train_model()
            agent.soft_update_target()
        avg_10_reward.append(total_reward)
        #  agent.write_summray(np.mean(avg_10_reward), actor_loss, critic_loss, total_step)

        if episode % 10 == 0:
            print(f"Episode {episode} | Total Steps {total_step} | Avg Reward: {np.mean(avg_10_reward):.2f} | Actor_Loss:{actor_loss:.4f} | Critic_Loss:{critic_loss:.4f}")

        # if TRAIN_MODE and episode % SAVE_INTERVAL == 0:
        #     agent.save_model()

    env.close()
