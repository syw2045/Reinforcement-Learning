import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0, x)

X = np.arange(-5.0, 5.0, 0.1)
Y = relu(X)
plt.plot(X, Y)
plt.ylim(-0.1, 5.5)
plt.show()