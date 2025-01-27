import numpy as np

x = np.array([1,2,3])
pi = np.array([0.1, 0.1, 0.8])

# 기대값의 참값 계산
e = np.sum(x * pi)
print('참값(E_pi[x]) :', e)

b = np.array([0.2, 0.2, 0.6])
n = 100
samples = []

for _ in range(n):
    idx = np.arange(len(b))
    i = np.random.choice(idx,p=b)
    s = x[i]
    rho = pi[i] / b[i]
    samples.append(rho * s)

mean = np.mean(samples)
var = np.var(samples)
print('중요도샘플링 : {:.2f} (분산 : {:.2f})'.format(mean, var))

