
def rawrelu(v):
	ret = {}
	for key in v.v:
		if v.v[key]<0:
			ret[key] = 0
		else:
			ret[key] = v.v[key]
	return rawvar(ret)

def derive(func, v):
	ret = {}
	if func==rawrelu:
		for key in v.v:
			if v.v[key]>0:
				ret[key]=1
			else:
				ret[key]=0
	return idmat(ret)



class idmat:
	def __init__(self, input):
		self.v = input
#it is the identical matrix of rawvar


class rawvar:
	def __init__(self, input):
		self.v = input

	def __add__(self, other):
		if type(other)!=rawvar:
			return self
		ret = {}
		for key in self.v:
			ret[key] = self.v[key]
		for key in other.v:
			if key in ret:
				ret[key]+=other.v[key]
			else:
				ret[key] = other.v[key]
		var = rawvar(ret)
		return var


	def __radd__(self, other):
		if type(other)!=rawvar:
			return self
		ret = {}
		for key in self.v:
			ret[key] = self.v[key]
		for key in other.v:
			if key in ret:
				ret[key]+=other.v[key]
			else:
				ret[key] = other.v[key]
		var = rawvar(ret)
		return var



	def __sub__(self, other):
		if type(other)!=rawvar:
			return self
		ret = {}
		for key in self.v:
			ret[key] = self.v[key]
		for key in other.v:
			if key in ret:
				ret[key]-=other.v[key]
			else:
				ret[key] = other.v[key]*-1
		var = rawvar(ret)
		return var

	def __rsub__(self, other):
		if type(other)!=rawvar:
			return self
		ret = {}
		for key in self.v:
			ret[key] = self.v[key]
		for key in other.v:
			if key in ret:
				ret[key]-=other.v[key]
			else:
				ret[key] = other.v[key]*-1
		var = rawvar(ret)
		return var



	def __mul__(self, other):
		ret = {}
		dot = 0

		if type(other)==rawvar:		
			for key in self.v:
				if key in other.v:
					dot += self.v[key]*other.v[key]
			return dot

		elif type(other)==idmat:
			for key in self.v:
				if key in other.v:
					ret[key] = self.v[key]*other.v[key] 
				else:
					ret[key]=0
			return rawvar(ret)

		else:
			for key in self.v:
				ret[key] = self.v[key]*other
			return rawvar(ret)

	def __rmul__(self, other):
		ret = {}
		dot = 0

		if type(other)==rawvar:		
			for key in self.v:
				if key in other.v:
					dot += self.v[key]*other.v[key]
			return dot

		elif type(other)==idmat:
			for key in self.v:
				if key in other.v:
					ret[key] = self.v[key]*other.v[key] 
				else:
					ret[key]=0
			return rawvar(ret)

		else:
			for key in self.v:
				ret[key] = self.v[key]*other
			return rawvar(ret)
