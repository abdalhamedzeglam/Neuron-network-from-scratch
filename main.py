import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nn import layers
from nn import activations
from nn import losses
from nn import metrics

nnfs.init()

x , y = spiral_data(100 , 3)

Layer1 = layers.Layer_Dence(2 , 3)
activation_layer1 = activations.ReLU_Activation()

Layer_ouput = layers.Layer_Dence(3 , 3)
activation_output = activations.Softmax_Activation()

Layer1.forward(x)
activation_layer1.forward(Layer1.output)

Layer_ouput.forward(activation_layer1.output)
activation_output.forward(Layer_ouput.output)

print(activation_output.output[:5])

loss_function = losses.Loss_Categorical_cross_entropy()
loss = loss_function.calculate_loss(activation_output.output , y)

metric = metrics.Accuracy()
accuracy = metric.calculate(activation_output.output , y)

print('loss:' , loss)
print('accuracy:' , accuracy , '%')