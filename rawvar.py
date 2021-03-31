
def rawrelu(v):
	pass



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
		if type(other)!=rawvar:
			for key in self.v:
				ret[key] = self.v[key]*other
			return rawvar(ret)
		elif type(other)==rawvar:		
			for key in self.v:
				if key in other.v:
					dot += self.v[key]*other.v[key]
			return dot

	def __rmul__(self, other):
		ret = {}
		dot = 0
		if type(other)!=rawvar:
			for key in self.v:
				ret[key] = self.v[key]*other
			return rawvar(ret)
		elif type(other)==rawvar:		
			for key in self.v:
				if key in other.v:
					dot += self.v[key]*other.v[key]
			return dot

