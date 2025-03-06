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

load_model = True
train_mode = False

batch_size = 32
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

# Model save and load path
date_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
save_path = f"./saved_models/Atari/Pong/DDQN/{date_time}"
load_path = "./saved_models/Atari/Pong/DDQN/250306150226"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDQN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DDQN, self).__init__(**kwargs)
        
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = torch.nn.BatchNorm2d(32)
        
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Flatten input size를 동적으로 계산하기 위해 임의의 데이터 통과시킴
        self._init_fc_input_size()

        self.fc1 = torch.nn.Linear(self.flatten_size, 512)
        
        # Dueling Architecture
        self.value_stream = torch.nn.Linear(512, 1)  
        self.advantage_stream = torch.nn.Linear(512, action_size)  

    def _init_fc_input_size(self):
        """CNN 출력 크기를 동적으로 계산하는 함수"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 64, 80)  # (batch, channels, height, width)
            dummy_out = self._forward_conv(dummy_input)
            self.flatten_size = dummy_out.view(1, -1).size(1)  # Flatten된 크기 저장

    def _forward_conv(self, x):
        """CNN 부분만 따로 forward 실행"""
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
    
if __name__ == '__main__':
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    agent = DDQNAgent(env)
    
    obs, info = env.reset() # (210, 160, 3) -> preprocessing (64, 80)
    obs = agent.process_observation(obs) # (64, 80)

    state = np.stack((obs, obs, obs, obs)) # (4, 64, 80)
    print(state.shape)
    action = agent.get_action(state)
    print(action)