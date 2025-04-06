import numpy as n
import matplotlib.pyplot as mp
x=[]
for i in range(10):
    x.append(n.random.uniform(-5,5))
x.sort()
yi=[]
def sig(x):
    return 1/(1+n.exp(-x))
for i in x:
    if i>0:
        yi.append(1)
    elif i<0:
        yi.append(0)
a=0.001    
x0=n.array([1,1,1,1,1,1,1,1,1,1])
x1=n.array(x)
y1=n.array(yi)
q=n.array([1.,1.])
X=n.array([x0,x1])
n1=10000
l=len(y1)
e=n.array(10)
for i in range(n1):
    z=X.T@q
    yp=sig(z)
    dj=X@(yp-y1)
    q-=a*dj
print(q)
mp.scatter(x,yi,label="Data Points")
x2=n.linspace(-5,5,100)
mp.plot(x2,sig(x2*q[1]+q[0]),label="Predicted Value")
mp.xlabel("X")
mp.ylabel("σ(x)")
mp.title("Log-Loss Cost Function")
mp.legend()
mp.show()
    
    
    
    

