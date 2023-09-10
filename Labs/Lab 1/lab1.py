import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.arange(5, 10, 0.05)

x[0:3]

print("The first three entries of x are:", x[0:3])

w = 10**(-np.linspace(1, 10, 10))

w

x = np.linspace(1, len(w), len(w))

plt.semilogy(x, w)
plt.show()

s = 3*w

plt.semilogy(x, w)
plt.semilogy(x, s)
plt.show()