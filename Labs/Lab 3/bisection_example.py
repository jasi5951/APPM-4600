# import libraries
import numpy as np

def driver():

# use routines    
    f = lambda x: x**2*(x-1)
    a1 = 0.5
    b1 = 2

    a2 = -1
    b2 = 0.5

    a3 = -1
    b3 = 2

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1


    # Problem 1 of lab
    tol = 1e-7

    [astar,ier] = bisection(f,a1,b1,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))

    [astar,ier] = bisection(f,a2,b2,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))

    [astar,ier] = bisection(f,a3,b3,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))


    # Problem 2 of lab
    tol = 1e-5
    f2 = lambda x: (x-1)*(x-3)*(x-5)
    a21 = 0
    b21 = 2.4

    [astar,ier] = bisection(f,a21,b21,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))

    f3 = lambda x: (x-1)**2*(x-3)
    a22 = 0
    b22 = 2

    [astar,ier] = bisection(f3,a22,b22,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))

    f4 = lambda x: np.sin(x)
    a23 = 0.5
    b23 = 3*np.pi/4

    [astar,ier] = bisection(f4,a23,b23,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    





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
    fb = f(b);
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
      
driver()     

