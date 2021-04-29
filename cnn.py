import numpy as np
from PIL import Image as pimg

def img2arr(name):
	img = pimg.open(name)
	img_arr = np.array(img)
	return img_arr

def avgpool_color(img, n):
#make smaller img array by putting each cell the average value of the cover respectively
#this method makes more precise pooling than maxpool does

	base = 255
	#this value is what pad value is

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
	for i in range(-1, st_r+1):
		cv = []
		for j in range(-1, st_c+1):
			dv = []
			for k in range(d):
				sum = 0
				for t1 in range(n):
					for t2 in range(n):
						x = i*n+leftpad+t1
						y = j*n+toppad+t2
						if x<0 or y<0 or x>=row or y>=col: 
							sum+=base
						else:
							sum+=img[x][y][k]
				sum = sum/n/n
				sum = round(sum)
				dv.append(sum)
			cv.append(dv)
		rv.append(cv)
	ret = np.array(rv)
	ret = ret.astype(np.uint8)
	return ret

def avgpool_mono(img, n):
#mono color img implementation

	base = 255
	#this value is what pad value is

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
			sum = 0
			for t1 in range(n):
				for t2 in range(n):
					x = i*n+leftpad+t1
					y = j*n+toppad+t2
					if x<0 or y<0 or x>=row or y>=col: 
						sum+=base
					else:
						sum+=img[x][y]
			sum = sum/n/n
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
	for i in range(-1, st_r+1):
		cv = []
		for j in range(-1, st_c+1):
			dv = []
			for k in range(d):
				cover = []
				for t1 in range(n):
					for t2 in range(n):
						x = i*n+leftpad+t1
						y = j*n+toppad+t2
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


test2 = img2arr("test2.png")
test2 = test2[:,:,1]
pooled = maxpool_mono(test2, 3)
print(test2.shape)
print(pooled.shape)
pooled = pimg.fromarray(pooled)
pooled.show()
