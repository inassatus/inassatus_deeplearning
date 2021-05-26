import random
import math
import rawvar
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
	if v>0:
		return math.log(v)
	elif v<0:
		return -1*math.log(-1*v)
	else:
		return 0

def derive(func, v):

	if func == linear:
		return 1
	elif func == relu:
		if v>0:
			return 1
		else:
			return 0
	elif func == ln:
		if v>0:
			return 1/v
		elif v<0:
			return 1/v
		else:
			return 0
	elif func == exp:
		return exp(v)
	else:
		if type(v)==rv:
			return rawvar.derive(func, v)


class neuron:
	def __init__(self):
		self.v =  0
		self.w = []
		self.mutrate = 0.003
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
			self.w[i] = random.gauss(self.w[i], self.mutrate/3)

	def clone(self):
		ret = neuron()
		ret.v = self.v
		for w in self.w:
			ret.w.append(w)
		ret.mutrate = self.mutrate
		ret.activation = self.activation
		return ret


class network:
	def __init__(self, net):
		self.net = []
		self.b = 1
		self.learn = -1.00
		self.err = 0
		self.size = net
		for n in net:
			layer = []
			for i in range(n):
				layer.append(neuron())
			self.net.append(layer)
		self.setw()
		self.addb()

	def setw(self):
		for i in range(len(self.net)-1):
			for n in self.net[i]:
				n.setw(len(self.net[i+1]))

	def setb(self, b):
		self.b = b 

	def addb(self):
		for i in range(1, len(self.net)-1):
			bias = neuron()
			bias.v = self.b
			bias.setw(len(self.net[i+1]))
			self.net[i].append(bias)

	def setlearn(self, n):
		self.learn = n

	def reset(self):
		for l in self.net:
			for x in l:
				x.reset()
		for i in range(1, len(self.net)-1):
			self.net[i][len(self.net[i])-1].v = self.b

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

	def setcushion(self):
		if len(self.net)<4:
			return False
		else:
			for x in self.net[0]:
				x.setactive(ln)
			for x in self.net[1]:
				x.setactive(exp)
			return True
		#this setting will allow the relation between inputs

	def revive(self):
		print("reviving process initiated")
		for x in self.net:
			for y in x:
				for z in y.w:
					if z==0:
						z=random.random()
		print("reviving process completed")

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
			if type(x.activ)==rv:
				print(x.activ.v)
			else:
				print(x.activ)			

	def geto(self):
		ret = []
		for x in self.net[-1]:
			if type(x.activ)==rv:
				ret.append(x.activ.v)
			else:
				ret.append(x.activ)
		return ret

	def geterr(self, fixdo):
		olderr = self.err
		self.err = 0
		ol = self.net[len(self.net)-1]
		for i in range(len(ol)):
			oerr = ol[i].activ-fixdo[i]
			self.err += oerr*oerr
		self.err /= len(ol)
	#get total error of the network

	def getinputgrad(self):
		ret = []
		for x in self.net[0]:
			ret.append(x.effect)
		return ret

	def setvariance(self, fixedo):
		ol = self.net[len(self.net)-1]
		for i in range(len(ol)):
			ol[i].effect = ol[i].activ-fixedo[i]
			ol[i].effect *= 2/len(ol)
			ol[i].effect *=  derive(ol[i].activation, ol[i].v)
		self.backpro(len(self.net)-2)
	#set the variance of output: how much the estimated output is different from the correct output

	def backpro(self, n):
		if n==-1:
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
			mag+=(x.effect*x.effect)

		for i in range(len(self.net)-1):
			for x in self.net[i]:
				for j in range(len(x.w)):
					dw = x.activ*self.net[i+1][j].effect
					mag+=(dw*dw)

		for i in range(1, len(self.net)-1):
			db = 0
			for j in range(len(self.net[i][-1].w)):
				db+=self.net[i][-1].w[j]*self.net[i+1][j].effect
			db*=derive(self.net[i][-1].activation, self.net[i][-1].v)
			mag+=(db*db)
		mag+=0
		return mag
	#find magnitude of the gradient vector. we want the unit change of error as 1, but de = dw^2 since we use gradient descending
	#so, we need to make it into dw^2/mag^2: return mag^2 here

	def modweights(self):
		mag = self.findunit()
		if mag==0 and self.err!=0:
			print("gradient reached zero")
			return False
		for i in range(len(self.net)-1):
			for x in self.net[i]:
				for j in range(len(x.w)):
					dw = x.activ*self.net[i+1][j].effect
					x.w[j]+=self.learn*dw/mag*self.err
					#Time complexity = O(log_n(self.err*10^max_floating_fix))
					#change of err per epoch = err(1-1/n)
					#n=number of weights
					#sadly, this algorithm has a problem of Zeno's paradox
					#it reaches to the minimum very quick, but it never actually catches the point

		return True

	def fixnet(self, fixt):
		self.geterr(fixt)
		if self.err==0:
			return
		self.setvariance(fixt)
		if not self.modweights():
			self.revive()

	def optimize(self, variable, observed):
		self.input(variable)
		self.fixnet(observed)

	def autolearn(self):
		pass

	def clone(self):
		ret = network(self.size)
		for i in range(len(self.net)):
			for j in range(len(self.net[i])):
				ret.net[i][j] = self.net[i][j].clone()
		ret.b = self.b
		ret.learn = self.learn
		return ret

