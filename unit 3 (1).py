import numpy as np
import matplotlib.pyplot as plt
x=np.array([65,63,67,64,68,62,70,66,68,67,69,71])
y=np.array([68,66,68,65,69,66,68,65,71,67,68,70])
error=np.array([1,2,5,3,2,1,6,2,4,3,1,4])
w=1/error**2
n=len(x)
t=1.812
plt.subplot(1,1,1)
a=(sum(x*x)*sum(y)-sum(x)*sum(x*y))/(n*sum(x**2)-sum(x)**2)
b=(n*sum(x*y)-sum(x)*sum(y))/(n*sum(x**2)-sum(x)**2)
plt.plot(x,b*x+a,label=f"Without error ({b:.3f},{a:.3f})")
Sx2=sum((x-sum(x)/n)**2/n)
Sy2=sum((y-sum(y)/n)**2/n)
Sxy=sum((x-sum(x)/n)*(y-sum(y)/n))/n
r=Sxy/(np.sqrt(Sx2*Sy2))
print(f"Slope:- {b} intercept:- {a}")
print(f"Correlation coefficient:- {r}")
yf=b*x+a
stdError=np.sqrt(sum((y-yf)**2)/n)
betaPlus=b+t*stdError/np.sqrt((n-2)*Sx2)
betaMinus=b-t*stdError/np.sqrt((n-2)*Sx2)
ypPlus=yf-t*stdError*np.sqrt(n+1+n*(x-sum(x)/n)**2/Sx2)/np.sqrt(n-2)
ypMinus=yf+t*stdError*np.sqrt(n+1+n*(x-sum(x)/n)**2/Sx2)/np.sqrt(n-2)
print(f"Beta plus: {betaPlus:.3f}, Beta Minus: {betaMinus:.3f}")
plt.scatter(x,ypPlus,label="yp Plus")
plt.scatter(x,ypMinus,label="yp Minus")
sortedIndex=np.argsort(x)
xArranged=x[sortedIndex]
yArranged=y[sortedIndex]
errorSorted=error[sortedIndex]
plt.errorbar(xArranged,yArranged,yerr=errorSorted)
plt.grid()
plt.legend()
print(stdError)
plt.show()
