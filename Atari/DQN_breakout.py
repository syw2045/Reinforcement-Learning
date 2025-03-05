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

# Hyperparameters
state_size = [4, 84, 84]  # (C, H, W)
action_size = 4  # Noop / Fire / Right / Left

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 50000
discount_factor = 0.99
learning_rate = 0.00025

run_step = 50000 if train_mode else 0
test_step = 5000
train_start_step = 5000
target_update_step = 500

print_interval = 10
save_interval = 100

epsilon_eval = 0.05
epsilon_init = 1.0 if train_mode else epsilon_eval
epsilon_min = 0.1
explore_step = run_step * 0.8
eplsilon_delta = (epsilon_init - epsilon_min)/explore_step if train_mode else 0

# Model save and load path
date_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
save_path = f"./saved_models/Atari/DQN/{date_time}"
load_path = "./saved_models/Atari/DQN/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN 
class DQN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.conv1 = torch.nn.Conv2d(in_channels=state_size[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((state_size[1] - 8)//4 + 1, (state_size[2] - 8)//4 + 1)
        
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)

        # self.flat = torch.nn.Flatten() # FC의 입력을 위해 1차원으로 변경
        self.fc1 = torch.nn.Linear(64*dim3[0]*dim3[1], 512)
        self.q = torch.nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.q(x)



# DQN_Agent
class DQNAgent:
    def __init__(self):
        self.network = DQN().to(device) # Network 생성
        self.target_network = copy.deepcopy(self.network) # Target Network 생성
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.epsilon = epsilon_init
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
    # Epsilon greedy
    def get_action(self, state, training=True):
        # print("stacked_state shape:", state.shape)
        self.network.train(training)
        epsilon = self.epsilon if training else epsilon_eval

        # 랜덤하게 action 선택
        if epsilon > random.random():  
            action = np.random.randint(0, action_size)
        else:
            q = self.network(torch.FloatTensor(state).to(device))
            action = torch.argmax(q, axis=-1, keepdim=True).data.cpu().numpy()
        
        return action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state      = np.stack([b[0] for b in batch], axis=0)
        action     = np.stack([b[1] for b in batch], axis=0)
        reward     = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done       = np.stack([b[4] for b in batch], axis=0)

        state      = torch.FloatTensor(state).to(device)  # (32, 4, 84, 84)
        next_state = torch.FloatTensor(next_state).to(device)  # (32, 4, 84, 84)
        
        action = torch.LongTensor(action).unsqueeze(1).to(device)  # (32,) → (32, 1)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)  # (32,) → (32, 1)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)  # (32,) → (32, 1)

        # Q-value 계산
        q_values = self.network(state)  # (32, 4) (각 액션에 대한 Q값)
        q = q_values.gather(1, action)  # 선택된 action의 Q값만 가져옴 → (32, 1)
        with torch.no_grad():
            next_q_values = self.target_network(next_state)  # (32, 4)
            max_next_q = next_q_values.max(1, keepdim=True)[0]  # (32, 1)
            target_q = reward + (1 - done) * discount_factor * max_next_q  # (32, 1)

        # 손실 계산
        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(epsilon_min, self.epsilon - eplsilon_delta)

        return loss.item()

    # target network update
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, save_path+'/ckpt')

    def write_summray(self, score, loss, epsilon, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/loss", loss, step)
        self.writer.add_scalar("model/epsilon", epsilon, step)

def preprocessing(state):
    gray_state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    resized_state = cv2.resize(gray_state, (84, 84))
    normalized_state = resized_state / 255.0
    return normalized_state.astype(np.float32)

if __name__ == '__main__':
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    state, _ = env.reset() # state : (210, 160, 3)
    frames = deque(maxlen=4)
    # Agent
    agent = DQNAgent()
    losses, scores, episode, score = [], [], 0, 0

    for _ in range(4):
        frames.append(preprocessing(state))

    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False

        stacked_state = np.stack(frames, axis=0)
       # print("stacked_state",stacked_state.shape)
        action = agent.get_action(stacked_state) # 에이전트의 행동 선택
        next_state, reward, done, _, _ = env.step(action)

        frames.append(preprocessing(next_state))
        state = preprocessing(next_state) # state 업데이트
        next_stacked_state = np.stack(frames, axis=0)
       # print("next_stacked_state" ,next_stacked_state.shape)
        score += reward
        if train_mode:
            agent.append_sample(stacked_state, action, reward, next_stacked_state, done)

        if train_mode and step > max(batch_size, train_start_step):
            # Training
            loss = agent.train_model()
            losses.append(loss)

            # Target Network update
            if step % target_update_step == 0:
                agent.update_target()

        if done:
            scores.append(score)
            score = 0
            env.reset()
            episode +=1
            

            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_loss = np.mean(losses)
                agent.write_summray(mean_score, mean_loss, agent.epsilon, step)
                losses, scores = [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +  f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")

            # Model save 
            if train_mode and episode % save_interval == 0:
                agent.save_model()

    env.close()