import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math



# Problem 1
f = lambda x: 1/(1+(10*x)**2)

# You can change this N
N = 18
xN = 18

xplot = np.zeros(xN)
h = 2/(xN-1)

for i in range(xN):
    xplot[i] = -1 + (i-1)*h

Vrow = lambda x: list(reversed([x**i for i in range(xN)]))

V = lambda x: np.array([Vrow(x[i]) for i in range(xN)])
x = np.linspace(-1, 1, xN)

Vinv = inv(V(x))
y = f(x)
c = np.matmul(y, Vinv)

#p = lambda x: np.dot(Vrow(x), c)

newx = np.linspace(-1, 1, N)

p = np.zeros(N)
for i in range(N):
    p[i] = np.dot(Vrow(newx[i]), c)

plt.figure()
plt.plot(xplot, f(xplot), 'o')
plt.plot(newx, f(newx), color="blue")
plt.plot(newx, p, color="red")
plt.show()



# Problem 2
N = 15
''' interval'''
a = -1
b = 1


''' create equispaced interpolation nodes'''
xint = np.linspace(a,b,N+1)

''' create interpolation data'''
yint = f(xint)

''' create points for evaluating the Lagrange interpolating polynomial'''
Neval = 1000
xeval = np.linspace(a,b,Neval+1)
yeval_l= np.zeros(Neval+1)
''' evaluate lagrange poly '''
for kk in range(Neval+1):
    yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)

fex = f(xeval)
       
 
plt.plot(xeval,fex,'ro-')
plt.plot(xeval,yeval_l,'bs--') 
plt.show()

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)



# Problem 3
N = 15
''' interval'''
a = -1
b = 1


''' create equispaced interpolation nodes'''
xint = np.linspace(a,b,N+1)

''' create interpolation data'''
yint = f(xint)

''' create points for evaluating the Lagrange interpolating polynomial'''
Neval = 1000
#xeval = np.linspace(a,b,Neval+1)
xeval = np.zeros(Neval+1)
for i in range(Neval + 1):
    xeval[i] = np.cos((2*i-1)*math.pi/(2*Neval+1))

yeval_l= np.zeros(Neval+1)
''' evaluate lagrange poly '''
for kk in range(Neval+1):
    yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)

fex = f(xeval)
       
 
plt.plot(xeval,fex,'ro-')
plt.plot(xeval,yeval_l,'bs--') 
plt.show()
