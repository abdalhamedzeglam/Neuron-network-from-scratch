import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dence:
    def __init__(self , N_inputs , N_neurons):
        self.weights = 0.01 * np.random.randn(N_inputs , N_neurons)
        self.biases = np.zeros((1 , N_neurons))

    def forward(self , inputs):
        self.output = np.dot(inputs , self.weights) + self.biases

class Activation_ReLU:
    def forward(self , inputs):
        self.output = np.maximum(0 , inputs)

class Activation_Softmax:
    def forward(self , inputs):
        exp_value = np.exp(inputs - np.max(inputs , axis=1 , keepdims = True))
        norm = exp_value / np.sum(exp_value , axis=1 , keepdims=True)

        self.output = norm

x , y = spiral_data(samples=100 , classes=3)

dence1 = Layer_Dence(2 , 3)
activation_ReLU1 = Activation_ReLU()

dence2 = Layer_Dence(3 , 3)
activation_softmax = Activation_Softmax()

dence1.forward(x)
activation_ReLU1.forward(dence1.output)

dence2.forward(activation_ReLU1.output)
activation_softmax.forward(dence2.output)

print(activation_softmax.output[:5])