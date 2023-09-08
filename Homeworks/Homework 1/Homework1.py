import numpy as np
import matplotlib.pyplot as plt
import math

# Problem 1
p1 = lambda x: x**9 -18*x**8 +144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 6408*x**2 +2304*x - 512
p2 = lambda x: (x-2)**9

x1 = np.arange(1.920, 2.080, 0.001)

plt.plot(x1, p1(x1))
plt.show()
plt.plot(x1, p2(x1))


# Problem 5 a
x1 = 10
x2 = 5

delx1 = 0.0000001
delx2 = 0.0000001

y = x1 - x2
x1t = x1 - delx1
x2t = x2 - delx2

yt = y + (delx1 - delx2)

# Problem 5 b
delta = []
for i in range(17):
    delta.append(10**(-16)*10**(i))

f1 = lambda x, delta: np.cos(x+delta) - np.cos(x)
f2 = lambda x, delta: -2*np.sin((2*x + delta)/2)*np.sin(delta/2)

pidifference = []
for i in delta:
    pidifference.append(f1(np.pi, i) - f2(np.pi, i))

tenToThe6difference = []
for i in delta:
    tenToThe6difference.append(f1(10**6, i) - f2(10**6, i))

plt.semilogx(delta, pidifference)
plt.show()

plt.semilogx(delta, tenToThe6difference)
plt.show()


# Problem 5 c
f3 = lambda x, delta: -delta*np.sin(x) - delta**2/2*np.cos((2*x + delta)/2)

pidifferencef3 = []
for i in delta:
    pidifferencef3.append(f1(np.pi, i) - f3(np.pi, i))

tenToThe6differencef3 = []
for i in delta:
    tenToThe6differencef3.append(f1(10**6, i) - f3(10**6, i))


plt.semilogx(delta, pidifferencef3)
plt.show()

plt.semilogx(delta, tenToThe6differencef3)
plt.show()

