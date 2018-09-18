#!/usr/bin/python3

import matplotlib

from matplotlib import pyplot as plt 

import numpy as np 

#each point is height,width and type (0,1)


data = [[3,   1.5, 1],
		[2,   1,   0 ],
		[4,   1.5, 1],
		[3,   1,   0],
		[3.5,.5,   1],
		[2,  .5,   0],
		[5.5, 1,   1],
		[1,   1,   0]]

mystry_flower=[5, 1]


#print(data[2][1])

#network

#    o flower type
#   / \ w1 , w2 , b
#   o  o  length,width



def sigmoid(x):
	return 1/(1+np.exp(-x))


##JUst to plot a sigmoid function 

T = np.linspace(-6,6,100)
#print(T)

Y = sigmoid(T)
#plt.plot(T, Y,c='r')

def sigmoid_p(x):
	return sigmoid(x) * (1 - sigmoid(x))

#plt.plot(T, sigmoid_p(T),c='b')

#plt.show()
#print("lenght=",len(data))


##scatter data
plt.axis([0,6,0,6])
plt.grid()

for i in range(len(data)):
	point= data[i]
    #print the given data 
	#print(point)
	#print(point[0])
	color = "r"
	if point[2] == 0:
		color = "b"
	plt.scatter(point[0], point[1],c=color)
plt.show()



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

	target = point[2]

	cost = np.square(pred - point[2])
	#print(point,cost)
	
	costs.append(cost)


	dcost_pred = 2 * (pred - target)
	dpred_dz = sigmoid_p(z)

	#partial derivative of w1//how the z changes with the only changes of w1
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

	
	if i % 100 ==0:
		cost_sum = 0
		for j in range(len(data)):
			point = data[ri]
			z = point[0] * w1 + point[1] * w2 + b 
			pred = sigmoid(z)
			cost_sum=cost_sum+np.square(pred-target)
		costs.append(cost_sum/len(data))

plt.plot(costs)
plt.show()


for i in range(len(data)):
	point= data[i]
	print(point)
	z=point[0]*w1+point[1]*w2+b
	pred=sigmoid(z)
	#print("pred:",pred)

z=mystry_flower[0] * w1 + mystry_flower[1]*w2 +b

pred=sigmoid(z)
print("predicted color of the mystry_flower :",pred)

if pred < 0.5 :
	print("the color is = blue")
else :
	print("the color is = red")


