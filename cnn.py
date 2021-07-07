import numpy as np
import random
from PIL import Image as pimg
import time
import threading
from numba import jit, cuda, njit


def img2arr(name):
	img = pimg.open(name)
	img_arr = np.array(img)
	return img_arr

def revimg(img):
	ret = np.zeros(img.shape, dtype = float)
	for i in range(len(img)):
		for j in range(len(img[0])):
			ret[i][j]=img[i][j]/255.0
			ret[i][j]-=1
			ret[i][j]*=-1

	return ret

@njit
def round(v):
	a = v
	b = int(v)
	c = a-b
	if c>=0.5:
		c=1
	else:
		c=0
	return b+c


def avgpool_color(img, n):
#make smaller img array by putting each cell the average value of the cover respectively
#this method makes more precise pooling than maxpool does

	row = len(img)
	st_r = round(row/n)
	#total stride count of row
	left_r = row%n
	leftpad = round(left_r/2)
	rightpad = left_r-leftpad
	#pad size of left and right side
	col = len(img[0])
	st_c = round(col/n)
	#total stride count of col
	left_c = col%n
	toppad = round(left_c/2)
	botpad = left_c-toppad
	#pad size of top and bottom side
	#==> the image is pooled into n time smaller image
	d = len(img[0][0])

	rv = []
	for i in range(0, st_r):
		cv = []
		for j in range(0, st_c):
			dv = []
			for k in range(d):
				sum = 0
				count = 0
				for t1 in range(n):
					for t2 in range(n):
						x = i*n-leftpad+t1
						y = j*n-toppad+t2
						if x<0 or y<0 or x>=row or y>=col: 
							sum+=0
						else:
							sum+=img[x][y][k]
							count+=1
				sum = sum/count
				sum = round(sum)
				dv.append(sum)
			cv.append(dv)
		rv.append(cv)
	ret = np.array(rv)
	ret = ret.astype(np.uint8)
	return ret

def avgpool_mono(img, n):
#mono color img implementation

	row = len(img)
	st_r = round(row/n)
	#total stride count of row
	left_r = row%n
	leftpad = round(left_r/2)
	rightpad = left_r-leftpad
	#pad size of left and right side
	col = len(img[0])
	st_c = round(col/n)
	#total stride count of col
	left_c = col%n
	toppad = round(left_c/2)
	botpad = left_c-toppad
	#pad size of top and bottom side
	#==> the image is pooled into n time smaller image

	rv = []
	for i in range(0, st_r):
		cv = []
		for j in range(0, st_c):
			sum = 0
			count=0
			for t1 in range(n):
				for t2 in range(n):
					x = i*n-leftpad+t1
					y = j*n-toppad+t2
					if x<0 or y<0 or x>=row or y>=col: 
						sum+=0
					else:
						sum+=img[x][y]
						count+=1
			sum = sum/count
			sum = round(sum)
			cv.append(sum)
		rv.append(cv)
	ret = np.array(rv)
	return ret

def maxpool_color(img, n):
#makes smaller img array by putting the maximum value of the cover repectively
#pooled image gets brighter since each cell get the highest rgb value.
#however, this method is still useful in some cases: it performs better functioning than avgpool in some specific situation
#such as: pooling the covolution img
#derivation is 0 when v is not the max value, and 1 when v is the max value (v is some random value in masked area)

	row = len(img)
	st_r = round(row/n)
	#total stride count of row
	left_r = row%n
	leftpad = round(left_r/2)
	rightpad = left_r-leftpad
	#pad size of left and right side
	col = len(img[0])
	st_c = round(col/n)
	#total stride count of col
	left_c = col%n
	toppad = round(left_c/2)
	botpad = left_c-toppad
	#pad size of top and bottom side
	#==> the image is pooled into n time smaller image
	d = len(img[0][0])

	rv = []
	for i in range(0, st_r):
		cv = []
		for j in range(0, st_c):
			dv = []
			for k in range(d):
				cover = []
				for t1 in range(n):
					for t2 in range(n):
						x = i*n-leftpad+t1
						y = j*n-toppad+t2
						if x<0 or y<0 or x>=row or y>=col: 
							cover.append(0)
						else:
							cover.append(img[x][y][k])
				dv.append(max(cover))
			cv.append(dv)
		rv.append(cv)
	ret = np.array(rv)
	ret = ret.astype(np.uint8)
	return ret

def maxpool_mono(img, n):
#mono color img implementation

	row = len(img)
	st_r = round(row/n)
	#total stride count of row
	left_r = row%n
	leftpad = round(left_r/2)
	rightpad = left_r-leftpad
	#pad size of left and right side
	col = len(img[0])
	st_c = round(col/n)
	#total stride count of col
	left_c = col%n
	toppad = round(left_c/2)
	botpad = left_c-toppad
	#pad size of top and bottom side
	#==> the image is pooled into n time smaller image

	rv = []
	for i in range(0, st_r):
		cv = []
		for j in range(0, st_c):
			cover = []
			for t1 in range(n):
				for t2 in range(n):
					x = i*n-leftpad+t1
					y = j*n-toppad+t2
					if x<0 or y<0 or x>=row or y>=col: 
						cover.append(0)
					else:
						cover.append(img[x][y])
			cover.append(0)
			cv.append(max(cover))
		rv.append(cv)
	ret = np.array(rv)
	return ret

def batchnorm(X):
	mean = np.sum(X)/len(X)
	shifted = X - mean
	sigma = shifted*shifted
	sigma = np.sum(sigma)
	sigma = sigma/len(X)
	x = shifted/np.sqrt(sigma+0.000001)
	return x

def dbn(X, effect):

	mean = np.sum(X)/len(X)
	shifted = X - mean
	sigma = shifted*shifted
	sigma = np.sum(sigma)
	sigma = sigma/len(X)
	ns = np.sqrt(sigma+0.000001)
	x = shifted/ns

	te = np.sum(effect)
	tex = effect*x
	tex = np.sum(tex)

	ret = []
	for i in range(len(X)):
		temp = len(X)*effect[i]
		temp -= te
		temp -= x[i]*tex
		temp /= (len(x)*ns)
		ret.append(temp)
	ret = np.array(ret)
	ret = ret.astype(float)

	return ret

@njit
def applyfilt(img, filter):
#accepts only mono image: need to divide rgb image into r-img, g-img, and b-img
#relu is the default activation
	row = len(img)
	col = len(img[0])
	xfilt = len(filter)
	yfilt = len(filter[0])
	rv = []
	for i in range(row):
		cv = []
		for j in range(col):
			sum = 0
			for x in range(xfilt):
				for y in range(yfilt):
					xcor = i-(xfilt-round(xfilt/2))+x
					ycor = j-(yfilt-round(yfilt/2))+y
					if xcor<0 or xcor>=row or ycor<0 or ycor>=col:
						sum+=0
					else:
						sum+=img[xcor][ycor]*filter[x][y]
			cv.append(sum)
		rv.append(cv)
	ret = np.array(rv)
	return ret

def reversefilt(filt):
	ret = []
	for i in range(len(filt)-1, -1, -1):
		row = []
		for j in range(len(filt[i])-1, -1, -1):
			row.append(filt[i][j])
		ret.append(row)
	ret = np.array(ret)
	return ret

def getdf(img, filt, effect):
	row = len(img)
	col = len(img[0])
	xfilt = len(filt)
	yfilt = len(filt[0])
	lpad = xfilt-round(xfilt/2)
	tpad = yfilt-round(yfilt/2)
	ret = np.zeros((xfilt, yfilt), dtype=float)

	for i in range(len(effect)):
		for j in range(len(effect[0])):
			if effect[i][j]!=0:
				for x in range(xfilt):
					for y in range(yfilt):
						xcor = i-lpad+x
						ycor = j-tpad+y
						if xcor<0 or xcor>=row or ycor<0 or ycor>=col:
							ret[x][y]+=0
						else:
							ret[x][y]+=img[xcor][ycor]*effect[i][j]
	return ret



class conv:
	def __init__(self, shape):
		self.v = np.zeros(shape, dtype=float)
		self.filt = []
		self.effect = None
		self.activ = None
		self.shape = shape
		self.end = False

	def setfilt(self, n, d):
		for i in range(n):
			fx = []
			for j in range(d):
				fy = []
				for k in range(d):
					fy.append(random.random())
				fx.append(fy)
			self.filt.append(fx)
		self.filt = np.array(self.filt)
		self.filt = self.filt.astype(float)


	def reset(self):
		self.v = np.zeros(self.shape, dtype=float)
		self.effect = None
		self.activ = None

	def activate(self):
		if self.end:
			self.activ = self.v
			return

		self.activ = maxpool_mono(self.v, 2)

	def derive(self):
		if self.end:
			return

		ret = np.zeros(self.shape, dtype=float)	
		n = 2	
		row = len(self.v)
		st_r = round(row/n)
		left_r = row%n
		leftpad = round(left_r/2)
		rightpad = left_r-leftpad
		col = len(self.v[0])
		st_c = round(col/n)
		left_c = col%n
		toppad = round(left_c/2)
		botpad = left_c-toppad
		padn = 0
		for i in range(0, st_r):
			for j in range(0, st_c):
				selected = False
				for t1 in range(n):
					for t2 in range(n):
						x = i*n-leftpad+t1
						y = j*n-toppad+t2
						if x<0 or y<0 or x>=row or y>=col: 
							padn+=1
						else:
							if self.v[x][y]<self.activ[i][j] or selected:
								ret[x][y] = 0
							else:
								ret[x][y] = self.effect[i][j]
								selected=True
		self.effect = ret



class fc:
#fully connected layer has only relu activation
	def __init__(self):
		self.v = 0
		self.activ = 0
		self.w = []
		self.effect = 0
		self.dropout = 0
		self.end = False

	def setw(self, n):
		for i in range(n):
			self.w.append(random.random())

	def setdrop(self, v):
		#i won't implement dropout algorithm for now. however, it could be added anywhen
		#random.random() < self.dropout => self.activ = 0
		self.dropout = v

	def reset(self):
		self.v = 0
		self.activ = 0
		self.effect = 0

	def activate(self):
		if self.end:
			self.activ = self.v
			return

		if self.v>0:
			self.activ = self.v
		else:
			self.activ = 0

	def derive(self):
		if self.end:
			self.effect*=1
			return

		if self.v>0:
			self.effect*=1
		else:
			self.effect*=0

class softmax:
	def __init__(self):
		self.v = 0
		self.activ = 0
		self.effect = 0

	def reset(self):
		self.v = 0
		self.activ = 0
		self.effect = 0

	def activate(self, activ):
		self.activ = activ

	def seteffect(self, fixed, m):
		self.effect = self.activ - fixed
		self.effect /= m
		#I changed the loss function into cross entropy

class cnn:
	def __init__(self, net, fcsize, smsize, sample, filtsize):
		self.net = []
		self.size = net
		self.filtsize = filtsize
		self.learn = -0.01
		self.fc = []
		self.fcsize = fcsize
		self.softmax = []
		self.smsize = smsize
		self.img = sample
		self.err = 0

		net.append(1)
		net.insert(0, len(sample[0][0]))
		
		xshape = len(sample)
		yshape = len(sample[0])
		
		layer = []
		for i in range(net[0]):
			temp = conv((xshape, yshape))
			temp.end = True
			layer.append(temp)
		self.net.append(layer)
		for i in range(1, len(net)-1):
			layer = []
			for j in range(net[i]):
				temp = conv((xshape, yshape))
				layer.append(temp)
			self.net.append(layer)
			xshape = round(xshape/2)
			yshape = round(yshape/2)		
		layer = []
		for i in range(net[-1]):
			temp = conv((xshape, yshape))
			temp.end = True
			layer.append(temp)
		self.net.append(layer)

		self.setfilt()

		self.addb()

		temp1 = len(self.img)
		temp2 = len(self.img[0])

		for i in range(len(self.size)-2):
			temp1 = round(temp1/2)
			temp2 = round(temp2/2)


		fcinput = temp1*temp2
		self.fcsize.insert(0, fcinput)
		
		for n in self.fcsize:
			layer = []
			for i in range(n):
				layer.append(fc())
			self.fc.append(layer)
		for x in self.fc[0]:
			x.end = True
		for x in self.fc[-1]:
			x.end = True

		for i in range(smsize):
			self.softmax.append(softmax())

		self.setw()
		self.addfcb()


	def setfilt(self):
		for i in range(len(self.net)-1):
			for n in self.net[i]:
				n.setfilt(len(self.net[i+1]), self.filtsize)

	def setw(self):
		for i in range(len(self.fc)-1):
			for n in self.fc[i]:
				n.setw(len(self.fc[i+1]))
		for n in self.fc[-1]:
			n.setw(self.smsize)

	def addb(self):
		for i in range(len(self.net)-1):
			bias = conv(self.net[i][0].shape)
			bias.v = np.ones(bias.shape, dtype=float)
			bias.setfilt(len(self.net[i+1]), 1)
			if i==0:
				bias.end=True
			self.net[i].append(bias)

	def addfcb(self):
		for i in range(len(self.fc)-1):
			bias = fc()
			bias.v = 1
			bias.setw(len(self.fc[i+1]))
			self.fc[i].append(bias)

		bias = fc()
		bias.v = 1
		bias.setw(self.smsize)
		self.fc[-1].append(bias)


	def reset(self):
		for l in self.net:
			for x in l:
				x.reset()
		for i in range(len(self.net)-1):
			self.net[i][-1].v = np.ones(self.net[i][-1].shape, dtype=float)

		for l in self.fc:
			for x in l:
				x.reset()
		for fc in self.fc:
			fc[-1].v = 1

		for x in self.softmax:
			x.reset()


	def input(self, img):
		self.reset()
		self.err = 0
		self.img = img
		for i in range(len(img[0][0])):
			x = img[:,:,i]
			x=x.flatten()
			x=batchnorm(x)
			x=x.reshape(len(img), len(img[0]))
			self.net[0][i].v = x

		for i in range(len(self.net)-1):
			for x in self.net[i]:
				x.activate()
				for j in range(len(x.filt)):
					self.net[i+1][j].v+=applyfilt(x.activ, x.filt[j])

		fcinput = self.net[-1][0].v
		fcinput = fcinput.flatten()
		fcinput = batchnorm(fcinput)

		for i in range(len(fcinput)):
			self.fc[0][i].v = fcinput[i]

		for i in range(len(self.fc)-1):
			for x in self.fc[i]:
				x.activate()
				for j in range(len(x.w)):
					self.fc[i+1][j].v += x.activ*x.w[j]
		for x in self.fc[-1]:
			x.activate()
			for i in range(len(x.w)):
				self.softmax[i].v += x.activ*x.w[i]

		out = []
		for x in self.softmax:
			out.append(x.v)
		out = out-np.max(out)
		eout = np.exp(out)
		esum = np.sum(eout)
		ev = eout/esum
		#values are subtracted by the highest value
		#this works because of the fraction: ex, e(0)/e(0)+e(1) = e(9)/e(9)+e(10)

		for i in range(self.smsize):
			self.softmax[i].activate(ev[i])


	def print(self):
		ret = []
		for x in self.softmax:
			ret.append(x.activ)
		print(ret)

	def geterr(self, fixed):
		#find the entropy
		self.err = 0
		for i in range(self.smsize):
			self.err += fixed[i]*np.log(self.softmax[i].activ)
		self.err *= -1/self.smsize


	def setentropy(self, fixed):
		#categorical cross entropy is implemented here: only exclusive classes(more than 3) will be learned
		#need to implement binary cross entropy for label classification
		for i in range(self.smsize):
			self.softmax[i].seteffect(fixed[i], self.smsize)

		for x in self.fc[-1]:
			for i in range(len(x.w)):
				x.effect+=x.w[i]*self.softmax[i].effect
			x.derive()
		self.backprofc(len(self.fc)-2)

		ec = []
		for x in self.fc[0]:
			ec.append(x.effect)
		ec.pop()
		fcinput = self.net[-1][0].v
		fcinput = fcinput.flatten()
		ec = dbn(fcinput, ec)

		ec=np.array(ec)
		ec=ec.astype(float)
		ec=ec.reshape(len(self.net[-1][0].v), len(self.net[-1][0].v[0]))
		self.net[-1][0].effect = ec

		self.backproconv(len(self.net)-2)

	def backprofc(self, n):
		if n == -1:
			return

		nlayer = self.fc[n]
		elayer = self.fc[n+1]
		for x in nlayer:
			for i in range(len(x.w)):
				x.effect+=x.w[i]*elayer[i].effect
			x.derive()

		self.backprofc(n-1)

	def backproconv(self, n):
		if n == -1:
			return

		nlayer = self.net[n]
		elayer = self.net[n+1]
		for x in nlayer:
			x.effect = np.zeros((len(x.activ), len(x.activ[0])), dtype = float)
			for i in range(len(x.filt)):
				x.effect+=applyfilt(elayer[i].effect, reversefilt(x.filt[i]))
			x.derive()

		self.backproconv(n-1)
		#algorithm here, the backpropagation, is exactly same as the algorithm that I used in deeplearning.py
		#the reason why I use reversed filter here is because the order in application of mask is reverse in effect array


	def modfilts(self):
		for i in range(len(self.net)-1):
			for x in self.net[i]:
				for j in range(len(x.filt)):
					df = getdf(x.activ, x.filt[j], self.net[i+1][j].effect)
					x.filt[j]+=(self.learn*df)

	def modweights(self):
		for i in range(len(self.fc)-1):
			for x in self.fc[i]:
				for j in range(len(x.w)):
					dw = x.activ*self.fc[i+1][j].effect
					x.w[j]+=self.learn*dw
		for x in self.fc[-1]:
			for i in range(len(x.w)):
				dw = x.activ*self.softmax[i].effect
				x.w[i]+=self.learn*dw

	def fixcnn(self, fixed):
		self.geterr(fixed)
		if self.err==0:
			return
		self.setentropy(fixed)
		self.modfilts()
		self.modweights()

	def optmodel(self, input, output):
		self.input(input)
		self.fixcnn(output)



r1=img2arr("rps/rock/rock01-000.png")
r1=avgpool_color(r1, 5)
r2=img2arr("rps/rock/rock01-001.png")
r2=avgpool_color(r2, 5)
r3=img2arr("rps/rock/rock01-002.png")
r3=avgpool_color(r3, 5)
p1=img2arr("rps/paper/paper01-000.png")
p1=avgpool_color(p1, 5)
p2=img2arr("rps/paper/paper01-001.png")
p2=avgpool_color(p2, 5)
p3=img2arr("rps/paper/paper01-002.png")
p3=avgpool_color(p3, 5)
s1=img2arr("rps/scissors/scissors01-000.png")
s1=avgpool_color(s1, 5)
s2=img2arr("rps/scissors/scissors01-001.png")
s2=avgpool_color(s2, 5)
s3=img2arr("rps/scissors/scissors01-002.png")
s3=avgpool_color(s3, 5)

r4=img2arr("rps/rock/rock01-003.png")
r4=avgpool_color(r4, 5)


print(r1.shape)
a = cnn([21, 31], [36, 21], 3, r1, 3)

for i in range(50):
	print(i,"th learning")
	a.input(r1)
	a.fixcnn([1, 0, 0])
	print(a.err)
	a.input(r2)
	a.fixcnn([1, 0, 0])
	a.input(r3)
	a.fixcnn([1, 0, 0])
	a.input(p1)
	a.fixcnn([0, 1, 0])
	print(a.err)
	a.input(p2)
	a.fixcnn([0, 1, 0])
	a.input(p3)
	a.fixcnn([0, 1, 0])
	a.input(s1)
	a.fixcnn([0, 0, 1])
	print(a.err)
	a.input(s2)
	a.fixcnn([0, 0, 1])
	a.input(s3)
	a.fixcnn([0, 0, 1])

print("err:", a.err)
a.input(r4)
a.print()

#pooled = pimg.fromarray(test2)
#pooled.show()
