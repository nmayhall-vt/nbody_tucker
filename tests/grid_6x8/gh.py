import numpy as np
import scipy 

J = np.loadtxt("j12.new4",float)
def printMat(a):
   rows = a.shape[0]
   cols = a.shape[1]
   for i in range(0,rows):
      for j in range(0,cols):
         print("% 4.3f  " %a[i,j]),
      print
   print
n = J.shape[0]
I = np.zeros((n,n))


A = np.bmat([J,I,I])
B = np.bmat([I,J,I])
C = np.bmat([I,I,J])
J2 = np.bmat('A;B;C')
printMat(J2)

k = []
for i in range(0,3*n):
    k.append(i)

k = np.array(k)
for i in range(0,3*n):
    print("% 6i  " %(k[i]+1)),
