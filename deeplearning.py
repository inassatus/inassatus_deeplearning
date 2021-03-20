import random

def relu(v):
	if v<0:
		return 0
	else:
		return v

def linear(v):
	return v

def derive(func, v):
	if func == linear:
		return 1
	elif func == relu:
		if v>0:
			return 1
		else:
			return 0


class var:
	def __init__(self, name="#n", n=1):
		self.v ={}
		self.v[name] = n

	def __add__(self, other):
		x = var()
		temp = {}
		for key in self.v:
			pass



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
		self.learn = 0.01
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
			print(x.activ)

	def setvariance(self, fixedo):
		ol = self.net[len(self.net)-1]
		for i in range(len(ol)):
			ol[i].effect = ol[i].activ-fixedo[i]
			ol[i].effect *= 2/len(ol)
			ol[i].effect *=  derive(ol[i].activation, ol[i].v)
		self.backpro(len(self.net)-2)


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

	def modweights(self):
		for i in range(len(self.net)-1):
			for x in self.net[i]:
				for j in range(len(x.w)):
					x.w[j]-=self.learn*x.activ*self.net[i+1][j].effect

	def fixnet(self, fixt):
		self.setvariance(fixt)
		self.modweights()

	def autolearningrate(self):
		pass



a = network([2,1])
for i in range(300):
	a.input([1,2])
	a.fixnet([3])
for i in range(300):
	a.input([3,4])
	a.fixnet([7])
for i in range(300):
	a.input([1,4])
	a.fixnet([5])

a.input([2,5])
