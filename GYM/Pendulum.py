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

SAVE_INTERVAL = 500
PRINT_INTERVAL = 10

GAMMA = 0.9
LAMDA = 0.9

EPOCHS = 10
CLIP_RATIO = 0.2

ACTOR_LR = 0.0003
CRITIC_LR = 0.0003

BATCH_SIZE = 32 
BUFFER_SIZE = 10

ROLLOUT_LEN = 3

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

        self.memory = []
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

    def append_data(self, transition):
        self.memory.append(transition)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
        mu, std = self.actor(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.item(), log_prob.item()
 
    def compute_advantages(self, data):
        adv_data = []
        for mini_batch in data:
            state, action, reward, next_state, prob, done = mini_batch
            with torch.no_grad():
                td_target = reward + GAMMA * self.critic(next_state) * done
                delta = td_target - self.critic(state)
            delta = delta.numpy()

            adv_list = []
            adv = 0
            
            for delta_t in delta[::-1]:
                adv = GAMMA * LAMDA * adv + delta_t[0]
                adv_list.append([adv])
            adv_list.reverse()
            adv = torch.tensor(np.array(adv_list), dtype=torch.float)
            adv_data.append((state, action, reward, next_state, done, prob, td_target, adv))
        return adv_data


    def make_batch(self):
        state_b, action_b, reward_b, next_state_b, prob_b, done_b = [], [], [], [], [], []
        data = []

        for i in range(BUFFER_SIZE):
            for j in range(BATCH_SIZE):
                rollout = self.memory.pop()
                state_list, action_list, reward_list, next_state_list, prob_list, done_list = [], [], [], [], [], []

                for transition in rollout:
                    state, action, reward, next_state, prob, done = transition
                    state_list.append(state)
                    action_list.append([action])
                    reward_list.append([reward])
                    next_state_list.append(next_state)
                    prob_list.append([prob])
                    done_list.append([done])

                state_b.append(state_list)
                action_b.append(action_list)
                reward_b.append(reward_list)
                next_state_b.append(next_state_list)
                prob_b.append(prob_list)
                done_b.append(done_list)

            mini_batch =  torch.tensor(np.array(state_b), dtype=torch.float), \
                            torch.tensor(np.array(action_b), dtype=torch.float), \
                            torch.tensor(np.array(reward_b), dtype=torch.float), \
                            torch.tensor(np.array(next_state_b), dtype=torch.float), \
                            torch.tensor(np.array(prob_b), dtype=torch.float), \
                            torch.tensor(np.array(done_b), dtype=torch.float)
            data.append(mini_batch)

        return data

    def update(self):
        if len(self.memory) == BATCH_SIZE * BUFFER_SIZE:
            data = self.make_batch()
            data = self.compute_advantages(data)

            for _ in range(EPOCHS):
                for mini_batch in data:
                    state, action, reward, next_state, done, old_prob, td_target, advantage = mini_batch

                    mu, std = self.actor.forward(state)
                    dist = torch.distributions.Normal(mu, std)
                    new_prob = dist.log_prob(action)
                    ratio = torch.exp(new_prob - old_prob)
                    clipped_ratio = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO)

                    # Actor Loss
                    surr1 = ratio * advantage
                    surr2 = clipped_ratio * advantage
                    actor_loss = -torch.min(surr1, surr2).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.actor_optimizer.step()

                    # Critic Loss
                    values = self.critic(state).squeeze()
                    td_target = reward.squeeze() + GAMMA * values.detach().squeeze()
                    critic_loss = F.smooth_l1_loss(values, td_target.detach())

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.critic_optimizer.step()

                return actor_loss.item(), critic_loss.item()
        else:
            return 0.0, 0.0
    

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
        actor_loss, critic_loss = 0, 0

        while not done and step_cnt < MAX_STEP:
            for _ in range(ROLLOUT_LEN):
                action, log_prob = agent.get_action(state)
                next_state, reward, done, _, _ = env.step([action])

                rollout.append((state, action, reward, next_state, log_prob, done))
               
                if len(rollout) == ROLLOUT_LEN:
                    agent.append_data(rollout)
                    rollout = []
                    break

                step_cnt += 1
                total_reward += reward
                state = next_state
                total_step += 1
            
            a_loss, c_loss = agent.update()
            actor_loss += a_loss
            critic_loss += c_loss

        avg_10_reward.append(total_reward)
        agent.write_summray(np.mean(avg_10_reward), actor_loss, critic_loss, total_step)

        if episode % PRINT_INTERVAL == 0:
            print(f"Episode {episode} | Avg_Reward: {np.mean(avg_10_reward):.2f} |Step:{total_step}| Actor Loss: {actor_loss:.6f} | Critic Loss: {critic_loss:.6f}")
        
        if TRAIN_MODE and episode % SAVE_INTERVAL == 0:
            agent.save_model()

    env.close()
