# import libraries
import numpy as np

def driver():

    # test functions
    f1 = lambda x: 1+0.5*np.sin(x)

    # fixed point is alpha1 = 1.4987....
    f2 = lambda x: 3+2*np.sin(x)

    #fixed point is alpha2 = 3.09...
    Nmax = 100
    tol = 1e-6

    # test f1 '''
    x0 = 0.0
    [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print('Error message reads:',ier)

    print("aitken: ", aitken(xstar, tol, Nmax))

    #test f2 '''
    x0 = 0.0
    [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f2(xstar):',f2(xstar))
    print('Error message reads:',ier)

    print("aitken: ", aitken(xstar, tol, Nmax))

# define routines
def fixedpt(f,x0,tol,Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    x = np.array([])
    count = 0

    while (count <Nmax):
        count = count + 1
        x1 = f(x0)
        x = np.append(x, x1)
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [x,ier]
        x0 = x1

    xstar = x1
    x = np.append(x, x1)
    ier = 1
    return [x, ier]


 ## 3.2
def aitken(x, tol, Nmax):
    p = np.array([])
    newp = x[0] - (x[0+1] - x[0])**2/(x[0+2] - 2*x[0+1] + x[0])
    p = np.append(p, newp)
    for i in range(1, len(x) - 2):
        newp = x[i] - (x[i+1] - x[i])**2/(x[i+2] - 2*x[i+1] + x[i])
        p = np.append(p, newp)
        if (abs(p[i] - p[i-1]) < tol):
            return p
        
    return p


driver()
