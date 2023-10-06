import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm



### Problem 1
# Part a
[xstar, ier,its] = SlackerNewton([1, 1], 1e-10, 1000)
print([xstar, ier,its])


# Part c
[xstar, ier,its] = Newton([1, 1], 1e-10, 1000)
print([xstar, ier,its])

#Problem 3 Part b
[xstar, ier,its] = problem3([1, 1, 1], 1e-10, 1000)
print([xstar, ier,its])

def SlackerNewton(x0,tol,Nmax):
    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
       
        if (norm(x1-x0) < tol):
            xstar = x1
            ier = 0
            return[xstar, ier,its]
           
        x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]   

def Newton(x0,tol,Nmax):
    for its in range(Nmax):
        J = evalJ(x0)
        Jinv = inv(J)
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
       
        if (norm(x1-x0) < tol):
            xstar = x1
            ier = 0
            return[xstar, ier,its]
           
        x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its] 

def evalF(x): 

    F = np.zeros(2)
    
    F[0] = 3*(x[0]**2) - x[1]**2
    F[1] = 3*x[0]*(x[1]**2) - x[0]**3 - 1

    return F

def evalJ(x): 
    J = np.array([[6*x[0], -2*x[1]], 
        [3*(x[1]**2) - 3*(x[0]**2), 6*x[0]*x[1]]])
    return J

def evald(x):
    d = evalf(x)/((2*x[0])**2 + (8*x[1])**2 + (8*x[2])**2)
    return d

def evalf(x):
    return (x[0]**2) + 4*(x[1]**2) + 4*(x[2]**2)-16

def partialfx(x):
    return 2*x[0]

def partialfy(x):
    return 8*x[1]

def partialfz(x):
    return 8*x[2]


def problem3(x0, tol, Nmax):
    for its in range(Nmax):
        d = evald(x0)
        f = evalf(x0)
        x1 = x0 - d*np.array([partialfx(x0), partialfy(x0), partialfz(x0)])
       
        if (norm(x1-x0) < tol):
            xstar = x1
            ier = 0
            return[xstar, ier,its]
           
        x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its] 