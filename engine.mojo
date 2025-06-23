import math
from memory import ArcPointer

@value
struct Value[dt: DType](EqualityComparable, Stringable):

	var data: ArcPointer[Scalar[dt]]
	var grad: ArcPointer[Scalar[dt]] 
	var _prev: List[Value[dt]]
	var _backward: fn(
		ArcPointer[Scalar[dt]],
		ArcPointer[Scalar[dt]],
		List[Value[dt]]
		) -> None

	def __init__(out self, data: Scalar[dt], _children: List[Value[dt]] = []):
		self.data = ArcPointer(data)
		self.grad = ArcPointer(Scalar[dt](0))
		self._prev = _children

		fn _backward(
		data: ArcPointer[Scalar[dt]],
		grad: ArcPointer[Scalar[dt]],
		_prev: List[Value[dt]]
		) -> None:
			pass

		self._backward = _backward

	@always_inline
	def __add__(self, other: Value[dt]) -> Self:
		out = Value[dt](self.data[] + other.data[], [self, other])

		fn _backward(
		data: ArcPointer[Scalar[dt]],
		grad: ArcPointer[Scalar[dt]],
		_prev: List[Value[dt]]
		) -> None:
			_prev[0].grad[] += grad[]
			_prev[1].grad[] += grad[]

		out._backward = _backward
		return out

	@always_inline
	def __mul__(self, other: Value[dt]) -> Self:
		out = Value[dt](self.data[] * other.data[], [self, other])

		fn _backward(
		data: ArcPointer[Scalar[dt]],
		grad: ArcPointer[Scalar[dt]],
		_prev: List[Value[dt]]
		) -> None:
			_prev[0].grad[] += _prev[1].data[] * grad[]
			_prev[1].grad[] += _prev[0].data[] * grad[]

		out._backward = _backward
		return out

	@always_inline
	def __pow__(self, other: Value[dt]) -> Self:
		out = Value[dt](self.data[] ** other.data[], [self, other])
		fn _backward(
		data: ArcPointer[Scalar[dt]],
		grad: ArcPointer[Scalar[dt]],
		_prev: List[Value[dt]]
		) -> None:
			_prev[0].grad[] += (_prev[1].data[] *  _prev[0].data[] ** (_prev[1].data[] - 1)) * grad[]
			_prev[1].grad[] =  0

		out._backward = _backward
		return out
	
	@always_inline
	def relu(self) -> Self:
		out = Value[dt](Scalar[dt](self.data[] >=0), [self])

		fn _backward(
		data: ArcPointer[Scalar[dt]],
		grad: ArcPointer[Scalar[dt]],
		_prev: List[Value[dt]]
		) -> None:
			_prev[0].grad[] += data[] * grad[]

		out._backward = _backward
		return out

	@staticmethod	
	def build_topo(
		v: Value[dt], 
		mut topo: List[Value[dt]], 
		mut visited: List[Value[dt]] 
		) -> None:
		if v not in visited:
			visited.append(v)
			for i in range(len(v._prev)):
				Self.build_topo(v._prev[i], topo, visited)
			topo.append(v)

	@always_inline
	def backward(mut self):
		topo = List[Value[dt]]()

		visited = List[Value[dt]]()
		Self.build_topo(self, topo, visited)
		self.grad[] = 1

		for i in range(len(topo)):
			v = topo[len(topo) - 1 - i]  
			v._backward(v.data,v.grad,v._prev)

	@always_inline
	fn __eq__(self, other: Value[dt]) -> Bool:
		return (self.data[] == other.data[]).reduce_and()

	@always_inline
	fn __ne__(self, other: Value[dt]) -> Bool:
		return not self.__eq__(other)

	@always_inline
	fn __str__(self) -> String:
		return String(self.data[])

	@always_inline
	def __neg__(self) -> Self:
		return self * Value[dt](-1)

	@always_inline
	def __radd__(self, other: Value[dt]) -> Self:
		return other + self

	@always_inline
	def __sub__(self, other: Value[dt]) -> Self:
		return self + (-other)

	@always_inline
	def __rsub__(self, other: Value[dt]) -> Self:
		return other + (-self)
	
	@always_inline
	def __rmul__(self, other: Value[dt]) -> Self:
		return other * self
	
	@always_inline
	def __truediv__(self, other: Value[dt]) -> Self:
		return self * (other ** Self(-1))

	@always_inline
	def __rtruediv__(self, other: Value[dt]) -> Self:
		return self * (other ** Self(-1))
