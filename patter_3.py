#!/usr/bin/python3

import matplotlib

from matplotlib import pyplot as plt 
import numpy as np 

#each point is number--first two is input and the third one is output


#output of the data is in the fourth column which is equal to the second column

data = [[0,   0,  0,  0],
		[0,   0,  1,  0],
		[1,   0,  0,  0],
		[0,   1,  0,  1],
		[1,   1,  0,  1],
		[0,   1,  1,  1]]

test_data=[1, 0 , 1 , 0]

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_p(x):
	return sigmoid(x) * (1 - sigmoid(x))


##training_loop
learning_rate = 0.2
costs = []

w1 =np.random.randn()
w2 =np.random.randn()
b =np.random.randn()


for i in range(50000):
	ri = np.random.randint(len(data))
	point=data[ri]

	#create random points from data 
	#print(point)

	#first positon (index-0) of each data 
	#print(point[2])
	

	z= point[0] * w1 + point[1] * w2 + b 
	#print(z)
	pred=sigmoid(z)
	#print(h)

	target = point[3]

	cost = np.square(pred - point[3])
	#print(point,cost)
	
	costs.append(cost)

    #differenciation of my cost to get minimum cost
    #how my cost changes when my prediction changes
	dcost_pred = 2 * (pred - target)

	#how my prediction change when the value of z changes
	dpred_dz = sigmoid_p(z)

	#partial derivative of w1
	#//how the z changes with the only changes of w1
	dz_dw1=point[0]

	#partial derivative of w2//how the z changes with the only changes of w2
	dz_dw2=point[1]
	
	#partial derivative of w1////how the b changes with the only changes of b
	dz_db=1

	#how the cost changes with the changes only of w1
	#how_the_cost_changes_with_respect_to_w1=
	dcost_dw1= dcost_pred * dpred_dz * dz_dw1

	#how the cost changes with the changes only of w2
	dcost_dw2= dcost_pred * dpred_dz * dz_dw2

	#how the cost changes with the only changes of b
	dcost_db= dcost_pred * dpred_dz * dz_db


	w1 = w1 - learning_rate * dcost_dw1
	w2 = w2 - learning_rate * dcost_dw2
	b = b - learning_rate * dcost_db

	#print(w1)

plt.plot(costs)
plt.show()

	
for i in range(len(data)):
	point= data[i]
	print(point)
	z=point[0]*w1+point[1]*w2+b
	pred=sigmoid(z)
	print("pred:",pred)

z=test_data[0] * w1 + test_data[1]*w2 +b

pred=sigmoid(z)


print("test_data:",pred)

orginal_output=test_data[3]

if pred <0.5:
	print("Orginal ouput =",orginal_output, " , predicted output = 0")
else :
	
	print("Orginal ouput =",orginal_output, " , predicted output = 1")