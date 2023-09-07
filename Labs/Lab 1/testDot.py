# Importing the libraries
import numpy as np
import numpy.linalg as la
import math

def driver():

     n = 100
     # creates a list from 0 to pi with 100 time steps
     x = np.linspace(0,np.pi,n)

# this is a function handle.  You can use it to define 
# functions instead of using a subroutine like you 
# have to in a true low level language.     
     f = lambda x: x**2 + 4*x + 2*np.exp(x)
     g = lambda x: 6*x**3 + 2*np.sin(x)

     y = f(x)
     w = g(x)

# evaluate the dot product of y and w     
     dp = dotProduct(y,w,n)

# print the output
     print('the dot product is : ', dp)

     return

# computes the dot product
def dotProduct(x,y,n):
# Loops over both lists and multiplies both elements and then adds them
     dp = 0.
     for j in range(n):
        dp = dp + x[j]*y[j]

     return dp  
     
driver()               
