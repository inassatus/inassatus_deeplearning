import numpy as np
import random
from deeplearning import network as nn
from PIL import Image as pimg
import time

def img2arr(name):
	img = pimg.open(name)
	img_arr = np.array(img)
	return img_arr

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
		for j in range(0, st_c0):
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
	ret = ret.astype(np.uint8)
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
	for i in range(-1, st_r+1):
		cv = []
		for j in range(-1, st_c+1):
			cover = []
			for t1 in range(n):
				for t2 in range(n):
					x = i*n+leftpad+t1
					y = j*n+toppad+t2
					if x<0 or y<0 or x>=row or y>=col: 
						cover.append(0)
					else:
						cover.append(img[x][y])
			cv.append(max(cover))
		rv.append(cv)
	ret = np.array(rv)
	ret = ret.astype(np.uint8)
	return ret

def applyfilt(img, filter):
#accepts only mono image: need to divide rgb image into r-img, g-img, and b-img
	row = len(img)
	col = len(img[0])
	xfilt = len(filter)
	yfilt = len(filter[0])
	img = img.astype(float)
	rv = []
	for i in range(0-(xfilt-round(xfilt/2)), row-1-round(xfilt/2-1)):
		cv = []
		for j in range(0-(yfilt-round(yfilt/2)), col-1-round(yfilt/2-1)):
			sum = 0
			for x in range(xfilt):
				for y in range(yfilt):
					xcor = i+x
					ycor = j+y
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
	return ret

def getdf(img, filt, effect):
	row = len(img)
	col = len(img[0])
	xfilt = len(filt)
	yfilt = len(filt[0])
	ret = np.zeros((xfilt, yfilt), float)
	a=0
	for i in range(0-(xfilt-round(xfilt/2)), row-1-round(xfilt/2-1)):
		b=0
		for j in range(0-(yfilt-round(yfilt/2)), col-1-round(yfilt/2-1)):
			for x in range(xfilt):
				for y in range(yfilt):
					xcor = i+x
					ycor = j+y
					if xcor<0 or xcor>=row or ycor<0 or ycor>=col:
						ret[x][y]+=0
					else:
						ret[x][y]+=img[xcor][ycor]*effect[a][b]
			b+=1
		a+=1
	return ret

	

class conv:
	def __init__(self):
		self.v = None
		self.filt = []
		self.effect = None

	def setfilt(self, n, d):
		for i in range(n):
			fx = []
			for j in range(d):
				fy = []
				for k in range(d):
					fy.append(random.random())
				fx.append(fy)
			self.filt.append(fx)


class cnn:
	def __init__(self, net, filtsize):
		self.net = []
		self.size = net
		self.err = 0
		self.filtsize = filtsize
		self.learn = -1.0
		self.judge = None
		self.img = None
		net.append(1)

		for n in net:
			layer = []
			for i in range(n):
				layer.append(conv())
			self.net.append(layer)
		self.setfilt()

	def setfilt(self):
		for i in range(len(self.net)-1):
			for n in self.net[i]:
				n.setfilt(len(self.net[i+1]), self.filtsize)

	def input(self, img):
		self.img = img
		for i in range(len(self.net[0])):
			self.net[0][i].v = img[:, :, i]
		for i in range(1, len(self.net)):
			for conv in self.net[i]:
				conv.v = np.zeros((len(img), len(img[0])), dtype=float)
		for i in range(len(self.net)-1):
			for x in self.net[i]:
				for j in range(len(x.filt)):
					self.net[i+1][j].v+=applyfilt(x.v, x.filt[j])
		
	def setjudge(self, size):
		fixedsize = len(self.img)*len(self.img[0])
		size.insert(0, fixedsize)
		self.judge = nn(size)
		print("judgement setting complete")

	def applyjudge(self):
		inputimg = self.net[-1][0].v
		inputimg = inputimg.flatten()
		self.judge.input(inputimg)

	def printjudge(self):
		self.judge.print()

	def fixjudge(self, fixed):
		self.judge.fixnet(fixed)
		self.geteffect()
		self.backpro(len(self.net)-2)
		self.modfilts()

	def geteffect(self):
		err = self.judge.getinputgrad()
		dimg=np.array(err)
		dimg = dimg.astype(float)
		dimg = dimg.reshape(len(self.img), len(self.img[0]))
		self.net[-1][0].effect = dimg
		
		self.err = 0
		for x in err:
			self.err+=(x*x)
		self.err /= len(err)

	def backpro(self, n):
		if n == -1:
			return

		nlayer = self.net[n]
		elayer = self.net[n+1]
		for x in nlayer:
			x.effect = np.zeros((len(self.img), len(self.img[0])), dtype = float)
			for i in range(len(x.filt)):
				x.effect+=applyfilt(elayer[i].effect, reversefilt(x.filt[i]))

		self.backpro(n-1)
		#algorithm here, the backpropagation, is exactly same as the algorithm that I used in deeplearning.py
		#the reason why I use reversed filter here is because the order in application of mask is reverse in effect array

	def findunit(self):
		mag = 0
		for e in self.net[0]:
			for x in e.effect:
				for y in x:
					mag+=(y*y)

		for i in range(len(self.net)-1):
			for x in self.net[i]:
				for j in range(len(x.filt)):
					df = getdf(x.v, x.filt[j], self.net[i+1][j].effect)
					for a in df:
						for b in a:
							mag+=(b*b)

		mag += 0
		return mag

	def modfilts(self):
		mag = self.findunit()

		for i in range(len(self.net)-1):
			for x in self.net[i]:
				for j in range(len(x.filt)):
					df = getdf(x.v, x.filt[j], self.net[i+1][j].effect)
					x.filt[j]+=self.learn*df/mag*self.judge.err



test2 = img2arr("test.jpg")
print(test2.shape)
test2 = avgpool_color(test2, 50)
print("pooled")
a = cnn([3, 3], 5)
a.input(test2)
a.setjudge([10, 12, 12, 1])
print("judgement set")

for i in range(50):
	a.input(test2)
	a.applyjudge()
	a.printjudge()
	a.fixjudge([1])
#pooled = pimg.fromarray(test2)
#pooled.show()
