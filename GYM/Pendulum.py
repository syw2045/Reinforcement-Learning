import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import random
import datetime
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Hyperparameter
STATE_DIM = 3
ACTION_DIM = 1
MAX_STEP = 200

UPDATE_INTERVAL = 200
SAVE_INTERVAL = 500
PRINT_INTERVAL = 10

GAMMA = 0.99
LAMDA = 0.95

EPOCHS = 10
CLIP_RATIO = 0.2

ACTOR_LR = 0.0001
CRITIC_LR = 0.0005

BATCH_SIZE = 32 
MEMORY_SIZE = 32

START_EPI = 1
END_EPI = 10000

TEST_MODE = False
TRAIN_MODE = True

# model save and load path
date_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
save_path = f"./saved_models/Pendulum/PPO/{date_time}"
load_path = f"./saved_models/Pendulum/PPO/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mu = nn.Linear(128, ACTION_DIM)
        self.fc_std = nn.Linear(128, ACTION_DIM)

    def forward(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value(x)


class PPOAgent:
    def __init__(self):
        self.actor = Actor().to(DEVICE)
        self.critic = Critic().to(DEVICE)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.memory = deque(maxlen=MEMORY_SIZE) 
        self.writer = SummaryWriter(save_path)

        if TEST_MODE == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=DEVICE)
            self.actor.load_state_dict(checkpoint["actor"])
            self.target_actor.load_state_dict(checkpoint["actor"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.target_critic.load_state_dict(checkpoint["critic"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
        mu, std = self.actor(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.item(), log_prob.item()
 
    def compute_advantages(self, rewards, values):
        advantages = torch.zeros_like(rewards).to(DEVICE)
        advantage = 0
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + GAMMA * values[t + 1] - values[t]
            advantage = delta + GAMMA * LAMDA * advantage
            advantages[t] = advantage
        return advantages

    def update(self):
        states, actions, rewards, next_states, old_log_probs, dones = zip(*random.sample(self.memory, BATCH_SIZE))

        states = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(DEVICE)
        old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32).to(DEVICE)

        for _ in range(EPOCHS):
            values = self.critic(states).squeeze()
            advantages = self.compute_advantages(rewards, values)

            mu, std = self.actor.forward(states, softmax_dim=1)
            dist = torch.distributions.Normal(mu, std)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO)

            # Actor Loss
            surr1 = ratio * advantages
            surr2 = clipped_ratio * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Critic Loss
            values = self.critic(states).squeeze()
            td_target = rewards + GAMMA * values.detach()
            critic_loss = F.smooth_l1_loss(values, td_target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            return actor_loss.item(), critic_loss.item()
    

    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "actor" : self.actor.state_dict(),
            "actor_optimizer" : self.actor_optimizer.state_dict(),
            "critic" : self.critic.state_dict(),
            "critic_optimizer" : self.critic_optimizer.state_dict(),
        }, save_path+'/ckpt')

    def write_summray(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)



if __name__ == "__main__":
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    agent = PPOAgent()
    avg_10_reward = deque(maxlen=10)
    rollout = []
    total_step = 0

    for episode in range(START_EPI, END_EPI):
        state, _ = env.reset()
        done = False
        step_cnt = 0
        total_reward = 0

        while not done and step_cnt < MAX_STEP:
            for _ in range(3):
                action, log_prob = agent.get_action(state)
                next_state, reward, done, _, _ = env.step([action])

                agent.memory.append((state, action, reward, next_state, log_prob, done))
               
            if len(agent.memory) >= BATCH_SIZE:
                actor_loss, critic_loss = agent.update()
                agent.memory.clear()

            step_cnt += 1
            total_reward += reward
            state = next_state
            total_step += 1

        avg_10_reward.append(total_reward)
        avg_reward = np.mean(avg_10_reward)
        
        agent.write_summray(total_reward, actor_loss, critic_loss, total_step)

        if episode % PRINT_INTERVAL == 0:
            print(f"Episode {episode} | Avg_Reward: {avg_reward:.2f} |Step:{total_step}| Actor Loss: {actor_loss:.2f} | Critic Loss: {critic_loss:.2f}")
        
        if TRAIN_MODE and episode % SAVE_INTERVAL == 0:
            agent.save_model()

    env.close()
