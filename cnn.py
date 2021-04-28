import numpy as np
from PIL import Image as pimg

def img2arr(name):
	img = pimg.open(name)
	img_arr = np.array(img)
	return img_arr

def avgpool_color(img, n):
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
							sum+=0
						else:
							sum+=img[x][y][k]
				sum = sum/n/n
				sum = round(sum)
				dv.append(sum)
			cv.append(dv)
		rv.append(cv)
	ret = np.array(rv)
	return ret




test2 = img2arr("test2.png")
pooled = avgpool(test2, 5)
print(test2.shape)
print(pooled.shape)
pooled = pimg.fromarray(pooled.astype(np.uint8))
pooled.show()
