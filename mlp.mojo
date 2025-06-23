from engine import Value
from random import random_float64

@value
struct Neuron[dt: DType]:
	var w: List[Value[dt]]
	var b: Value[dt]
	var nin: Int
	var nonlin: Bool
	def __init__(out self, nin: Int, nonlin: Bool=True):
		self.nin = nin
		self.nonlin = nonlin
		self.w = List[Value[dt]]()
		for i in range(self.nin):
			self.w.append(Value[dt](
						Scalar[dt](random_float64())
					)
				)
		self.b = Value[dt](0)

	def __call__(self, x:List[Value[dt]]) -> Value[dt]:
		out = Value[dt](0)
		for i in range(self.nin):
			out = out + (x[i] * self.w[i])

		return out.relu() if self.nonlin else out

	def zero_grad(self) -> None:
		for v in self.w:
			v.grad[] = 0

	def parameters(self) -> List[Value[dt]]:
		return self.w + [self.b]

@value
struct Layer[dt: DType]:
	var neurons: List[Neuron[dt]]
	var nin: Int
	var nout: Int
	var nonlin: Bool
	def __init__(out self, nin: Int, nout: Int, nonlin: Bool):
		self.neurons = List[Neuron[dt]]()
		self.nin = nin
		self.nout = nout
		self.nonlin = nonlin

		for i in range(self.nout):
			self.neurons.append(Neuron[dt](self.nin, self.nonlin))

	def __call__(self, x:List[Value[dt]]) -> List[Value[dt]]:
		out = List[Value[dt]]()
		for i in range(self.nout):
			out.append(self.neurons[i](x))
		return out

	def zero_grad(self) -> None:
		for i in range(self.nout):
			self.neurons[i].zero_grad()

	def parameters(self) -> List[Value[dt]]:
		out =	List[Value[dt]]()
		for i in range(self.nout):
			out += self.neurons[i].parameters()
		return out

struct MLP[dt: DType]:
	var layers: List[Layer[dt]]
	var n_layers: List[Int]
	def __init__(out self, nin: Int, nouts: List[Int]):
		self.n_layers = [nin] + nouts
		self.layers = List[Layer[dt]]()
		for i in range(1, len(self.n_layers)):
			nonlin = True if i == len(self.n_layers) else False
			self.layers.append(
				Layer[dt](
					self.n_layers[i - 1], 
					self.n_layers[i], 
					nonlin
					)
				)
	def __call__(self, x:List[Value[dt]]) -> List[Value[dt]]:
		var out: List[Value[dt]] = []
		for i in range(len(self.layers)):
			out = self.layers[i](x)
		return out

	def zero_grad(self):
		for i in range(len(self.layers)):
			self.layers[i].zero_grad()

	def parameters(self) -> List[Value[dt]]:
		out = List[Value[dt]]()
		for i in range(len(self.layers)):
			out += self.layers[i].parameters()
		return out


		





