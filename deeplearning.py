import random
import math
from rawvar import rawvar as rv
from math import exp

def relu(v):
	if v<0:
		return 0
	else:
		return v

def linear(v):
	return v

def ln(v):
	return math.log(v)

def derive(func, v):
	if func == linear:
		return 1
	elif func == relu:
		if v>0:
			return 1
		else:
			return 0
	elif func == ln:
		return 1/v
	elif func == exp:
		return exp(v)

class neuron:
	def __init__(self):
		self.v =  0
		self.w = []
		self.mutrate = 0.001
		self.activation = linear
		self.activ = 0
		self.effect = 0

	def reset(self):
		self.v = 0
		self.activ = 0
		self.effect = 0

	def setw(self, n):
		for i in range(n):
			self.w.append(random.random())

	def setmutrate(self, v):
		self.mutrate = v

	def setactive(self, func):
		self.activation = func

	def activate(self):
		self.activ = self.activation(self.v)

	def mutate(self):
		for i in range(len(self.w)):
			self.w[i] = self.w[i]+random.uniform(-1*self.w[i]*self.mutrate, self.w[i]*self.mutrate)


class network:
	def __init__(self, net):
		self.net = []
		self.b = 0
		self.learn = 1.05
		self.err = 0
		for n in net:
			layer = []
			for i in range(n):
				layer.append(neuron())
			self.net.append(layer)
		self.setw()

	def setw(self):
		for i in range(len(self.net)-1):
			for n in self.net[i]:
				n.setw(len(self.net[i+1]))

	def setb(self, b):
		self.b = b 

	def setlearn(self, n):
		self.learn = n

	def reset(self):
		for l in self.net:
			for x in l:
				x.reset()

	def setmutrate(self, n):
		for l in self.net:
			for x in l:
				x.setmutrate(n)

	def mutate(self):
		for l in self.net:
			for x in l:
				x.mutate()

	def setactive(self, func):
		for i in range(1, len(self.net)):
			for x in self.net[i]:
				x.setactive(func)

	def input(self, input):
		self.reset()
		for n in range(len(input)):
			self.net[0][n].v = input[n]

		for i in range(len(self.net)-1):
			for x in self.net[i]:
				x.activate()
				for j in range(len(x.w)):
					self.net[i+1][j].v+=x.activ*x.w[j]
		self.output()

	def output(self):
		for x in self.net[len(self.net)-1]:
			x.activate()

	def print(self):
		for x in self.net[len(self.net)-1]:
			print(x.activ)

	def printrv(self):
		for x in self.net[len(self.net)-1]:
			print(x.activ.v)

	def geterr(self, fixdo):
		olderr = self.err
		self.err = 0
		ol = self.net[len(self.net)-1]
		for i in range(len(ol)):
			oerr = ol[i].activ-fixdo[i]
			self.err += oerr*oerr
		self.err /= len(ol)
	#get total error of the network

	def setvariance(self, fixedo):
		ol = self.net[len(self.net)-1]
		for i in range(len(ol)):
			ol[i].effect = ol[i].activ-fixedo[i]
			ol[i].effect *= 2/len(ol)
			ol[i].effect *=  derive(ol[i].activation, ol[i].v)
		self.backpro(len(self.net)-2)
	#set the variance of output: how much the estimated output is different from the correct output

	def backpro(self, n):
		if n==0:
			return

		nlayer = self.net[n]
		elayer = self.net[n+1]
		for x in nlayer:
			x.effect = 0
			for i in range(len(x.w)):
				x.effect+=x.w[i]*elayer[i].effect
			x.effect*=derive(x.activation, x.v)

		self.backpro(n-1)
	#backpropagation is done here. Calculate how much each perceptron has effect on the variance

	def findunit(self):
		mag = 0
		for x in self.net[0]:
			dx = 0
			for i in range(len(x.w)):
				dx+=x.w[i]*self.net[1][i].effect
			dx*=derive(x.activation, x.v)

			mag+=(dx*dx)

		for i in range(len(self.net)-1):
			for x in self.net[i]:
				for j in range(len(x.w)):
					dw = x.activ*self.net[i+1][j].effect
					mag+=(dw*dw)
		mag+=0
		return mag
	#find magnitude of the gradient vector. we want the unit change of error as 1, but de = dw^2 since we use gradient descending
	#so, we need to make it into dw^2/mag^2: return mag^2 here

	def modweights(self):
		mag = self.findunit()
		if mag==0 and self.err!=0:
			print("local minimum entered")
			return False
		for i in range(len(self.net)-1):
			for x in self.net[i]:
				for j in range(len(x.w)):
					dw = x.activ*self.net[i+1][j].effect
					x.w[j]-=self.learn*dw/mag*self.err
					#Time complexity = O(log_n(self.err))
					#change of err per epoch = err/n
					#n=number of weights
		return True

	def fixnet(self, fixt):
		self.geterr(fixt)
		if self.err==0:
			return
		self.setvariance(fixt)
		self.modweights()

	def autolearn(self):
		pass

	def generation(self):
		pass


a = network([2,4,3,4,1])

for i in range(500):
	a.input([rv({"K":1, "Q":2}), rv({"K":2, "Q":3})])
	a.fixnet([rv({"K":3, "Q":-1})])

print(a.err)
a.input([rv({"K":1, "Q":2}), rv({"K":2, "Q":3})])
a.printrv()