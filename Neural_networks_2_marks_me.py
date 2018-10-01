#!/usr/bin/python3



import numpy as np

# x = (hours sleeping, hours studying), y = score on test
x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
xPredicted = np.array(([4,8]), dtype=float)

# scale units
X = x / np.amax(x) # maximum of X array
Y = y/100 # max test score is 10
xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)

#print(xPredicted)
#print(X)
#print(Y)

class Neural_Network(object):
	def __init__(self):
		self.inputSize = 2
		self.outputSize = 1
		self.hiddenSize = 3

		#3x2 input matrix X
		
		self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
		#(2x3) weight matrix from input to hidden layer
	
		self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 
		# (3x1) weight matrix from hidden to output layer

	def forward(self, X):
		#dot multiplication between input value and weight = z2
		self.z2= np.dot(X, self.W1)
		##3x2 dot 2x3 = 3x3
		
		#apply activation function to our total z2
		self.a2 = self.sigmoid(self.z2)
		#3x3

		#hidden layer to output layer ,dot multiplication between activation and weights  
		self.z3= np.dot(self.a2 , self.W2)
		# 3x3 dot 3x1 = 3x1

		#final activation function
		yhat = self.sigmoid(self.z3)
		#3x1 matrix
		
		return yhat

	def sigmoid(self, s):
		#activation function
		return 1/(1+np.exp(-s))

	def sigmoidPrime(self, s):
		#derivative of sigmoid
		return s * (1-s)

	def costfunction(self,X,Y):
		self.yhat = self.forward(X)
		j = 0.5 * sum((y- self.yhat)**2)
		return j

	def backward(self, X ,Y, yhat):

		self.o_error = Y - yhat # error in output 
		# Y = (3x1) and yhat = 3x1   

		self.delta3= self.o_error*self.sigmoidPrime(yhat)
		#3x1

		self.W2 +=np.dot(self.z2.T,self.delta3)
		#3x3 dot 3x1 = 3x1 matrix
		#print(self.W2)

		self.delta2 = np.dot(self.delta3,self.W2.T)*self.sigmoidPrime(self.z2)
		#3x3 matrix 
		#print(self.djdw1)

		self.W1 += np.dot(X.T,self.delta2)
		#3x2 matrix
		#print(self.W1)
	

		
	def trains(self,X,Y):
		yhat= self.forward(X)
		self.backward(X,Y,yhat)

	def predict(self):
		print("predicted data based on trained weights")
		print ("Input (scaled): \n" + str(xPredicted))
		print ("Output: \n" + str(self.forward(xPredicted)))

	
NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
   print ("# " + str(i) + "\n")
   print ("Input (scaled): \n" + str(X))
   print ("Actual Output: \n" + str(y))
   print ("Predicted Output: \n" + str(NN.forward(X)))
   print ("Loss: \n" + str(NN.costfunction(X,Y)))
   print ("\n")
   NN.trains(X,Y)
  

NN.predict()


