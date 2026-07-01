import numpy as np

class Layer_Dence:
    def __init__(self , N_inputs , N_neurons):
        self.weights = 0.01 * np.random.randn(N_inputs , N_neurons)
        self.biases = np.zeros((1 , N_neurons))
    
    def forward(self , inputs):
        self.output = np.dot(inputs , self.weights) + self.biases