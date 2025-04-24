import scipy.linalg
import numpy as np
import random
state = ["1", "2","3"]
MyMatrix = np.array([[0,1/2,1/2], [1/3,1/6,1/2],[1/2,1/4,1/4]])
n = 10
StartingState = random.randint(0,2)
CurrentState = StartingState
print(state[CurrentState], "--->", end=" ")
while n-1:
	CurrentState = np.random.choice([0,1,2], p=MyMatrix[CurrentState])
	print(state[CurrentState], "--->", end=" ")
	n -= 1
print("stop")
MyValues, left = scipy.linalg.eig(MyMatrix, right=False, left=True)
print('Transition matrix\n',MyMatrix)
print("left eigen vectors = \n", left, "\n")
print("eigen values = \n", MyValues)
pi = left[:, 0]
pi_normalized = [(x/np.sum(pi)).real for x in pi]
print('unnormalised',pi)
print('normalized',pi_normalized)

