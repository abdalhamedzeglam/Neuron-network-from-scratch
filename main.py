import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nn import layers
from nn import activations

nnfs.init()

x , y = spiral_data(100 , 3)

Layer1 = layers.Layer_Dence(2 , 3)
activations_layer1 = activations.ReLU_Activation()

Layer_ouput = layers.Layer_Dence(3 , 3)
activations_output = activations.Softmax_Activation()

Layer1.forward(x)
activations_layer1.forward(Layer1.output)

Layer_ouput.forward(activations_layer1.output)
activations_output.forward(Layer_ouput.output)

print(activations_output.output[:5])