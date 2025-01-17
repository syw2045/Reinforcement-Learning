import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.gridworld import GridWorld
from collections import defaultdict
import numpy as np

def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action:base_prob for action in range(action_size)}
    # 이 시점에서 action_probs는 {0:ε/4, 1:ε/4, 2:ε/4, 3:ε/4} 이 된다.
    action_probs[max_action] += (1-epsilon) #위에서 초기화를 해줬으니 += 가 가능하다.   
    return action_probs
 
class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1 # ε-탐욕 정책의 ε
        self.alpha = 0.1 # Q 함수 갱신 시의 고정값 α
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        # self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state] # 해당 state의 action probs
        actions = list(action_probs.keys()) 
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)
    
    def reset(self):
        self.memory.clear()
    
    def update(self):
        G = 0
        for data in reversed(self.memory):
            state,action,reward = data
            G = reward + self.gamma*G
            key = (state, action)
            # self.cnts[key] += 1
            self.Q[key] += (G - self.Q[key]) * self.alpha

            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)

env = GridWorld()
agent = McAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)

        if done:
            agent.update()
            break
        state = next_state

env.render_q(agent.Q)