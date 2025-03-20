import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import datetime
from collections import deque

from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
STATE_DIM = 3
ACTION_DIM = 1

GAMMA = 0.99
LAMDA = 0.95

CLIP_RATIO = 0.2
ACTOR_LR = 2e-4
CRITIC_LR = 2e-4

BATCH_SIZE = 32
EPOCHS = 10
ROLLOUT_LEN = 3

MAX_EPISODES = 10000
MAX_STEPS = 200

TEST_MODE = True
TRAIN_MODE = False

SAVE_INTERVAL = 300

date_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
save_path = f"./saved_models/Pendulum/PPO/{date_time}"
load_path = f"./saved_models/Pendulum/PPO/250319145246"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mu = nn.Linear(128, ACTION_DIM)
        self.fc_std = nn.Linear(128, ACTION_DIM)

    def forward(self, x):
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

        self.memory = []
        self.writer = SummaryWriter(save_path)

        if TEST_MODE == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=DEVICE)
            self.actor.load_state_dict(checkpoint["actor"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        mu, std = self.actor(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        return action.item(), dist.log_prob(action).sum().item()

    def append_data(self, transition):
        self.memory.append(transition)

    def train_model(self):
        if len(self.memory) < BATCH_SIZE:
            return 0,0

        batch = list(zip(*self.memory))
        state = torch.tensor(np.array(batch[0]), dtype=torch.float32, device=DEVICE)
        action = torch.tensor(np.array(batch[1]), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        reward = torch.tensor(np.array(batch[2]), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_state = torch.tensor(np.array(batch[3]), dtype=torch.float32, device=DEVICE)
        old_prob = torch.tensor(np.array(batch[4]), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        done = torch.tensor(np.array(batch[5]), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        self.memory.clear()

        with torch.no_grad():
            target = reward + GAMMA * self.critic.forward(next_state) * (1 - done)
            advantage = target - self.critic.forward(state)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(EPOCHS):
            indices = np.random.permutation(len(state))
            for i in range(0, len(state), BATCH_SIZE):
                batch_idx = indices[i:i+BATCH_SIZE]
                states, actions, advantages, targets, old_probs = state[batch_idx], action[batch_idx], advantage[batch_idx], target[batch_idx], old_prob[batch_idx]
                
                # Update Actor
                mu, std = self.actor.forward(states)
                dist = torch.distributions.Normal(mu, std)
                new_log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                ratio = torch.exp(new_log_prob - old_probs)
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Update Critic
                critic_loss = F.mse_loss(self.critic.forward(states), targets.detach())
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        return actor_loss, critic_loss

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
    env = gym.make("Pendulum-v1", render_mode="human")
    agent = PPOAgent()
    reward_history = deque(maxlen=10)
    total_step = 0

    for episode in range(1, MAX_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        actor_loss, critic_loss = 0, 0

        done = False
        while not done:
            action, log_prob = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step([action])
            done = terminated | truncated

            agent.append_data((state, action, reward, next_state, log_prob, done))
            state = next_state
            total_reward += reward
            total_step += 1
        
        if TRAIN_MODE:
            a_loss, c_loss = agent.train_model()
            actor_loss += a_loss
            critic_loss += c_loss
        
        reward_history.append(total_reward)
        

        agent.write_summray(np.mean(reward_history), actor_loss, critic_loss, total_step)

        if episode % 10 == 0:
            print(f"Episode {episode} | Total Steps {total_step} | Avg Reward: {np.mean(reward_history):.2f} | Actor_Loss:{actor_loss:.4f} | Critic_Loss:{critic_loss:.4f}")

        if TRAIN_MODE and episode % SAVE_INTERVAL == 0:
                    agent.save_model()

    env.close()

