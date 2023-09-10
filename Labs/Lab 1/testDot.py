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
     f1 = lambda x: np.cos(x)
     g1 = lambda x: np.sin(x)

     y = f1(x)
     w = g1(x)

# evaluate the dot product of y and w     
     dp = dotProduct(y,w,n)

# print the output
     print('the dot product is : ', dp)
     print("Numpy's matrix multiplication is: ", np.dot(w, y))
     arr1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
     arr2 = arr1
     mm = matMult(arr1, arr2)

     print("the matrix multiplication is: \n", mm)
    
     print("Numpy's matrix multiplication is:\n", np.matmul(arr1, arr2))

     return

# computes the dot product
def dotProduct(x,y,n):
# Loops over both lists and multiplies both elements and then adds them
     dp = 0.
     for j in range(n):
        dp = dp + x[j]*y[j]

     return dp  

def matMult(x, y):
    x = np.array(x)
    y = np.array(y)
    rows = np.shape(x)[0]
    columns = np.shape(y)[0]
    newMat = np.ndarray(shape = (rows, columns))
    for row in range(rows):
        for col in range(columns):
            dp = dotProduct(x[row, :], y[:, col], rows)
            newMat[row, col] = dp

    return newMat
     
driver()               
