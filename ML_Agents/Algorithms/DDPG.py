import numpy as np
import random
import copy
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

state_size = 9
action_size = 3

load_model = True
train_mode = False

batch_size = 128
mem_maxlen = 30000
discount_factor = 0.9
actor_lr = 1e-4
critic_lr = 5e-4
tau = 1e-3

mu = 0
theta = 1e-3
sigma = 2e-3

run_step = 50000 if train_mode else 0
test_step = 10000
train_start_step = 5000

print_interval = 10
save_interval = 100

game = "Drone"
env_name = f"../envs/{game}/{game}.exe"

# model save and load path
date_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
save_path = f"./saved_models/{game}/DDPG/{date_time}"
load_path = f"./saved_models/{game}/DDPG/250303160834"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OU_noise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X = np.ones((1, action_size), dtype=np.float32) * mu

    def sample(self):
        dx = theta * (mu - self.X) + sigma * np.random.randn(len(self.X))
        self.X += dx
        return self.X

class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.mu = torch.nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.mu(x))

class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128+action_size, 128)
        self.q = torch.nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.cat((x, action), dim=-1)
        x = torch.relu(self.fc2(x))
        return self.q(x)

class DDPGAgent():
    def __init__(self):
        self.actor = Actor().to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic().to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.OU = OU_noise()
        self.memory = deque(maxlen=mem_maxlen)
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.target_actor.load_state_dict(checkpoint["actor"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.target_critic.load_state_dict(checkpoint["critic"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def get_action(self, state, training=True):
        self.actor.train(training)

        action = self.actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
        return action + self.OU.sample() if training else action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state      = np.stack([b[0] for b in batch], axis=0)
        action     = np.stack([b[1] for b in batch], axis=0)
        reward     = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done       = np.stack([b[4] for b in batch], axis=0)

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                        [state, action, reward, next_state, done])

        next_actions = self.target_actor(next_state)
        next_q = self.target_critic(next_state, next_actions)
        target_q = reward + (1 - done) * discount_factor * next_q
        q = self.critic(state, action)
        critic_loss = F.mse_loss(target_q, q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_pred = self.actor(state)
        actor_loss = -self.critic(state, action_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def soft_update_target(self):
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

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


if __name__ == '__main__':
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name, side_channels=[engine_configuration_channel])
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
    dec, term = env.get_steps(behavior_name)

    agent = DDPGAgent()

    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        state = dec.obs[0]
        action = agent.get_action(state, train_mode)
        action_tuple = ActionTuple()
        action_tuple.add_continuous(action)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name)
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        next_state = term.obs[0] if done else dec.obs[0]
        score += reward[0]

        if train_mode:
            agent.append_sample(state[0], action[0], reward, next_state[0], [done])

        if train_mode and step > max(batch_size, train_start_step):
            actor_loss, critic_loss = agent.train_model()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            agent.soft_update_target()

        if done:
            episode += 1
            scores.append(score)
            score = 0

            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_actor_loss = np.mean(actor_losses)
                mean_critic_loss = np.mean(critic_losses)
                agent.write_summray(mean_score, mean_actor_loss, mean_critic_loss, step)
                actor_losses, critic_losses, scores = [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")

            if train_mode and episode % save_interval == 0:
                agent.save_model()

    env.close()