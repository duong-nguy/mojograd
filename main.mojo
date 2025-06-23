from engine import Value
from mlp import MLP



def loss_fn[dt: DType](y: Value[dt], y_hat: Value[dt]) -> Value[dt]:
	loss = (y - y_hat) ** Value[dt](2)
	return loss
	
def main():

	X = [
		[Value[DType.float32](0), Value[DType.float32](0)],
		[Value[DType.float32](0), Value[DType.float32](1)],
		[Value[DType.float32](1), Value[DType.float32](0)],
		[Value[DType.float32](1), Value[DType.float32](1)],
	]

	Y = [
		Value[DType.float32](0),
		Value[DType.float32](1),
		Value[DType.float32](1),
		Value[DType.float32](0),
	]
	
	lr = Scalar[DType.float32](0.1) 
	epochs = 1
	model = MLP[DType.float32](2,[2,1])
	avg_loss =Scalar[DType.float32](0.5) 
	for e in range(epochs):
		for i in range(len(X)):
			model.zero_grad()
			y_hat = model(X[i])
			loss = loss_fn[DType.float32](Y[i], y_hat[0])
			loss.backward()
			for p in model.parameters():
				p.data[] -= lr * p.grad[]
				print(p.grad[])
			avg_loss = loss.data[]
		print("Epoch:",e,"Loss:",avg_loss / 4)
		

