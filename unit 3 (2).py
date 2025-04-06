import numpy as n
import matplotlib.pyplot as mp
x0=n.array([1,1,1,1,1])
x1=n.array([1.1,2,3,4,5])
x2=n.array([2,4,6,8,10])
y=n.array([6.8,12.3,16.6,22.7,27.1])
q=n.array([1.,1.,1.])
n1=100
m=len(y)
x=n.array([x0,x1,x2])
x=x.T
a=0.001
j=[]
ind=[]
for i in range(n1):
    h=n.dot(x,q)
    e=y-h
    q+=(a/m)*(n.dot(x.T,e))
    j.append(n.sum(e**2)/m)
    ind.append(i)
print(q)
mp.plot(ind,j)
mp.show()



        
