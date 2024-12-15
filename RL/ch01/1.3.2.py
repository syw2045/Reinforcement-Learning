import numpy as np

np.random.seed(0) # 시드 고정
rewards = []
Q2 = 0

for n in range(1,11): # 10번 플레이
    reward = np.random.rand() # reward 무작위
    rewards.append(reward)
    
    Q1 = sum(rewards) / n
    Q2 += (reward - Q2) / n # n : 학습률(learning rate)

    print(Q1)
    print(Q2)
