import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import math


# Problem 1
Ti = 20
Ts = -15
alpha = 0.138e-6

f = lambda x: erf(x/(2*math.sqrt(alpha*60*60*60*24)))*(Ti-Ts) + Ts
fprime = lambda x: (Ti - Ts)/(math.sqrt(alpha*60*60*60*24*math.pi))*np.exp(-(x/(2*math.sqrt(alpha*60*60*60*24)))**2)


x = np.linspace(0, 5, 100)
plt.plot(x, f(x))
plt.show()

# Problem 1 b
# The answer my code prints is [0.6769615411758423, 0]
# So the root is 0.6769615411758423
print(bisection(f, 0, 5, 1e-6))


# Problem 1 c
# The answer my code prints is [[0.01, 0.6425178685850312, 0.6766921341272338, 0.6769618372769741, 0.6769618544819365], 0.6769618544819365, 0, 3]
# So the root is 0.6769618544819365
print(newton(f, fprime, 0.01, 1e-6, 1000))


# Problem 1 c
# My code breaks if I set my initial guess to x_bar most likely cause we divide by 0 because the slope is 0.
print(newton(f, fprime, 5, 1e-6, 1000))




# Problem 5
f = lambda x: x**6 - x - 1
fprime = lambda x: 6*(x**5) - 1


[iterations, root, ier, iteration] = newton(f, fprime, 2, 1e-6, 1000)
error = abs(np.array(iterations)-root)
errork1 = error[1:]
errork1 = np.append(errork1, 0)
plt.loglog(error, errork1)
plt.show()

[iterations, root, ier] = secant(f, 2, 1, 1e-6, 1000)
error = abs(np.array(iterations)-root)
errork1 = error[1:]
errork1 = np.append(errork1, 0)
plt.loglog(error, errork1)
plt.show()

# define routines
def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier]


def newton(f,fp,p0,tol,Nmax):
    """
    Newton iteration.
    
    Inputs:
        f,fp - function and derivative
        p0   - initial guess for root
        tol  - iteration stops when p_n,p_{n+1} are within tol
        Nmax - max number of iterations
    Returns:
        p     - an array of the iterates
        pstar - the last iterate
        info  - success message
            - 0 if we met tol
            - 1 if we hit Nmax iterations (fail)
        
    """
    p = [p0]
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p.append(p1)
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]



def secant(f, x0, x1, tol, Nmax):
    if (f(x0) == 0):
        return [x0, 0]
    if (f(x1) == 0):
        return [x1, 0]
    p = []
    p.append(x0)
    p.append(x1)
    fx1 = f(x1)
    fx0 = f(x0)
    for j in range(Nmax):
        if (abs(fx1-fx0) == 0):
            print("Divide by 0, bad")
            return [p, x1, 1]
        x2 = x1 - f(x1)*(x1-x0)/(f(x1) - f(x0))
        p.append(x2)
        if (abs(x2-x1) < tol):
            return [p, x2, 0]
        x0 = x1
        fx0 = fx1
        x1 = x2
        fx1 = f(x2)
    return [p, x2, 1]