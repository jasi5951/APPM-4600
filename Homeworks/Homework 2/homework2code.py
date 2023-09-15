import numpy as np
import matplotlib.pyplot as plt
import math




#Problem 2 of Homework
A = 1/2*np.array([[1, 1], [1+1e-10, 1-1e-10]])
b = np.array([[1], [1]])
x = np.array([[1], [1]])
invA = np.array([[1-1e10, 1e10], [1+1e10, -1e10]])
delb = np.array([[1e-5], [-1e-5]])
delbsame = np.array([[1e-5], [1e-5]])

relerror1 = np.linalg.norm(np.matmul(invA, delb), 2) / np.linalg.norm(x, 2)
relerror2 = np.linalg.norm(np.matmul(invA, delbsame), 2) / np.linalg.norm(x, 2)





# Problem 3 of Homework
y = lambda x: np.exp(x) - 1
ynew = lambda x: np.exp(x)
ynewfinal = lambda x: x - 1

x = np.linspace(-100, 100, 1000)
plt.plot(x, y(x))
plt.title("x vs f(x) plot")
plt.show()

plt.plot(x, ynewfinal(ynew(x)) - 1)
plt.title("x vs algorithm plot")
plt.show()

plt.plot(x, (ynewfinal(ynew(x))) - (y(x)))
plt.title("error between algorithm and f(x)")
plt.show()



print(y(9.999999995000000e-10))
print(ynewfinal(ynew(9.999999995000000e-10)))


tol = 1e-10
xstar = 9.999999995000000e-10

taylor = lambda x: 1 + x + x**2/2

fx = 1e-9

error = abs(fx - taylor(xstar))/abs(fx)
print(error <= tol)






# Problem 4 of Homework
t = np.arange(0, np.pi, np.pi/30)

y = lambda t: np.cos(t)

N = len(t)
S = 0
for k in range(N):
    S = S + t[k]*y(k)

print("The sum is: ", S)




R = 1.2
delr = 0.1
f = 15
p = 0
x = lambda z: R*(1 + delr*np.sin(f*z + p))*np.cos(z)
y = lambda z: R*(1 + delr*np.sin(f*z + p))*np.sin(z)

theta = np.linspace(0, 2*np.pi, 1000)
xtheta = x(theta)
ytheta = y(theta)

plt.plot(xtheta, ytheta)
plt.show()



for i in range(1, 11):
    R = i
    f = 2 + i
    delr = 0.05
    p = np.random.uniform(0, 2)
    xtheta = x(theta)
    ytheta = y(theta)
    plt.plot(xtheta, ytheta)

plt.show()