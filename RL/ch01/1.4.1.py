import numpy as np

class Bandit:
    def __init__(self, arms=10): # arms = 슬롯머신 대수
        self.rates = np.random.rand(arms) # 슬롯머신 각각의 승률 설정(무작위)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand(): # 슬롯머신의 승률 > 랜덤 승률
            return 1
        else :
            return 0
        
bandit = Bandit()

for i in range(3):
    print(bandit.play(0))
    