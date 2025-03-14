import numpy as np
from dezero import Variable

def rosenbrock(x0, x1):
    y = 100 * ( x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

Iters = 10000
Lr = 0.001

for i in range(Iters):
    print(x0,x1)
    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()

    y.backward()

    x0.data -= Lr * x0.grad.data
    x1.data -= Lr * x1.grad.data
print(x0, x1)