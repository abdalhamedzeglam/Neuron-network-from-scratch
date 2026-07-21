import nnfs
from nnfs.datasets import spiral_data

from nn import Layer_dense
from nn import ReLU
from nn import Softmax_CCE_loss
from nn import Accuracy

nnfs.init()

x , y = spiral_data(samples=100 , classes=3)

dense1 = Layer_dense(2 , 3)
relu = ReLU()

dense2 = Layer_dense(3 , 3)
loss_activation = Softmax_CCE_loss()

dense1.forward(x)
relu.forward(dense1.output)

dense2.forward(relu.output)
loss = loss_activation.forward(dense2.output , y)

accuracy_nn = Accuracy()
accuracy = accuracy_nn.calculate(loss_activation.output , y)

print(loss_activation.output[:5])
print('loss' , loss)
print('accuracy' , accuracy)

loss_activation.backward(loss_activation.output , y)
dense2.backward(loss_activation.dinputs)
relu.backward(dense2.dinputs)
dense1.backward(relu.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)