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

bandit = Bandit()
Qs = np.zeros(10) # 각 슬롯머신의 가치 추정치
ns = np.zeros(10) # 각 슬롯머신의 플레이 횟수

for n in range(10): # 10번 반복
    action = np.random.randint(0, 10) # 0~9 임의의 슬롯머신 선택
    reward = bandit.play(action) # 슬롯머신 플레이

    ns[action] += 1 # 플레이 횟수 증가
    Qs[action] += (reward - Qs[action]) / ns[action] # 증분 구현
    print(Qs)