import gym
import numpy as np
import datetime
import torch
import torch.nn.functional as F
import copy
import random
import cv2

from collections import deque
from torch.utils.tensorboard import SummaryWriter

action_size = 6

load_model = False
train_mode = True

# batch_size = 32
mem_maxlen = 500000
discount_factor = 0.98
learning_rate = 0.0001

run_step = 1000000 if train_mode else 0
test_step = 50000
train_start_step = 50000
target_update_step = 10000 # update 주기

print_interval = 10
save_interval = 100

epsilon_eval = 0.05
epsilon_init = 1.0 if train_mode else epsilon_eval
epsilon_min = 0.01
# explore_step = run_step * 0.8
explore_step = run_step
eplsilon_delta = (epsilon_init - epsilon_min)/explore_step if train_mode else 0


BATCH_SIZE = 64  # Minibatch size that select randomly from mem for train nets
MAX_EPISODE = 100000  # Max episode
MAX_STEP = 100000  # Max step size for one episode

MAX_MEMORY_LEN = 50000  # Max memory len
MIN_MEMORY_LEN = 40000  # Min memory len before start train


# Model save and load path
date_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
save_path = f"./saved_models/Atari/Pong/DDQN/{date_time}"
load_path = "./saved_models/Atari/Pong/DDQN/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDQN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DDQN, self).__init__(**kwargs)
        
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = torch.nn.BatchNorm2d(32)
        
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        self.fc_input_size()

        self.fc1 = torch.nn.Linear(self.flatten_size, 512)
        
        # Dueling Architecture
        self.value_stream = torch.nn.Linear(512, 1)  
        self.advantage_stream = torch.nn.Linear(512, action_size)  

    def fc_input_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 64, 80)  # (batch, channels, height, width)
            dummy_out = self._forward_conv(dummy_input)
            self.flatten_size = dummy_out.view(1, -1).size(1)  # Flatten된 크기 저장

    def _forward_conv(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


    
class DDQNAgent:
    def __init__(self, env):
        self.network = DDQN().to(DEVICE) # Network 생성
        self.target_network = copy.deepcopy(self.network) # Target Network 생성
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.epsilon = epsilon_init
        self.memory = deque(maxlen=mem_maxlen)
        self.epsilon = epsilon_init
        self.writer = SummaryWriter(save_path)

        if load_model == True:
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
            action = random.randrange(action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
                print("get_action : ", state.shape) # torch.Size([1, 4, 64, 80])
                q_values = self.network.forward(state)  # (1, action_size)
                action = torch.argmax(q_values).item()  # Returns the indices of the maximum value of all elements
        return action
    

    def train_model(self):
        if len(agent.memory) < MIN_MEMORY_LEN:
            loss, max_q = [0, 0]
            return loss, max_q
        # We get out minibatch and turn it to numpy array
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

        return loss, torch.max(state_q_values).item()

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append([state[None, :], action, reward, next_state[None, :], done])



if __name__ == '__main__':
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    agent = DDQNAgent(env)
    
    obs, info = env.reset() # (210, 160, 3) -> preprocessing (64, 80)
    obs = agent.process_observation(obs) # (64, 80)

    state = np.stack((obs, obs, obs, obs)) # (4, 64, 80)
    

    total_max_q_val = 0  # Total max q vals
    total_reward = 0  # Total reward for each episode
    total_loss = 0  # Total loss for each episode
    

    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False

        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        next_state = agent.process_observation(next_state)  # Process image
        next_state = np.stack((next_state, state[0], state[1], state[2]))

        agent.append_sample(state, action, reward, next_state, done)

        state = next_state

        if train_mode:
            loss, max_q_val = agent.train_model()  # Train with random BATCH_SIZE state taken from mem
        else:
            loss, max_q_val = [0, 0]

        total_loss += loss
        total_max_q_val += max_q_val
        total_reward += reward
        total_step += 1

