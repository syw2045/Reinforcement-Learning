import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mu, std

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value(x)

class PPO:
    def __init__(self, state_dim, action_dim, gamma=0.99, clip_ratio=0.2, lr=0.0001):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.buffer = []

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        mu, std = self.actor(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        return action.clamp(-1, 1).detach().numpy()

    def store(self, transition):
        self.buffer.append(transition)

    def compute_advantages(self, rewards, values, next_value, dones, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * lam * gae * (1 - dones[i])
            advantages.insert(0, gae)
            next_value = values[i]
        return torch.tensor(advantages, dtype=torch.float32)


    def update(self):
        states, actions, rewards, dones, old_log_probs = zip(*self.buffer)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32)

        values = self.critic(states).squeeze()
        advantages = self.compute_advantages(rewards, values, values[-1].item(), dones)

        mu, std = self.actor(states)
        dist = torch.distributions.Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        value_loss = F.mse_loss(self.critic(states).squeeze(), rewards + self.gamma * values.detach())
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        self.buffer = []
        return policy_loss, value_loss



reward_history = []
batch_size = 512

if __name__ == "__main__":
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(state_dim, action_dim)

    max_steps = env.spec.max_episode_steps  # max Steps

    for episode in range(10000):
        state, _ = env.reset()
        done = False
        step_count = 0
        total_reward = 0  
        while not done and step_count < max_steps:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(2.0 * action)
            agent.store((state, action, reward, done, np.log(1e-8 + np.abs(action))))
            state = next_state
            step_count += 1
            total_reward += reward 

        reward_history.append(total_reward)  

        if len(agent.buffer) >= batch_size:
            policy_loss, value_loss = agent.update()
            agent.buffer = []

        else: policy_loss, value_loss = 0,0

        if episode % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | actor_loss: {policy_loss:.2f} | critic_loss:{value_loss:.2f}")