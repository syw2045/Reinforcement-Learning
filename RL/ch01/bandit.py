import numpy as np

class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0
        
class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward): # 슬롯머신 가치 추정
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action] # 증분 구현

    def get_action(self):
        if np.random.rand() < self.epsilon: # 탐색(exploration) : self.epsilon의 확률로 아무 슬롯머신 play
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs) # 활용(exploitation) : 탐욕정책 -> 가장 좋다고 생각하는 슬롯머신 play
                                  # argmax : 가장 큰 원소(가치가 가장 큰) 의 인덱스 가져옴


import matplotlib.pyplot as plt

steps = 1000
epsilon = 0.1

bandit = Bandit()
agent = Agent(epsilon)
total_reward = 0
total_rewards = [] # 보상 합
rates = []

for step in range(steps):
    action = agent.get_action() # Agent가 action(슬롯머신) 선택
    reward = bandit.play(action) # 선택한 슬롯머신 Play -> reward
    agent.update(action, reward) # Action, reward 관계 학습
    total_reward += reward

    total_rewards.append(total_reward)
    rates.append(total_reward / (step + 1))

print(total_reward)

plt.ylabel('Total reward')
plt.xlabel('Steps')
plt.plot(total_rewards)
plt.show()

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(rates)
plt.show()