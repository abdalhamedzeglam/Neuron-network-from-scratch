import numpy as np
from .losses import categorical_cross_entropy

class ReLU():
    def forward(self , inputs):
        self.inputs = inputs
        self.output = np.maximum(0 , inputs)

    def backward(self , dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Softmax():
    def forward(self , inputs):
        self.inputs = inputs

        exp_value = np.exp(inputs - np.max(inputs , axis=1 , keepdims=1))

        probabilities = exp_value / np.sum(exp_value , axis=1 , keepdims=1)
        self.ouptut = probabilities

    def backward(self , dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_ouptut , single_dvalues) in \
                enumerate(zip(self.output , dvalues)):
            single_output = single_output.reshape(-1 , 1)

            jacobian_matrix = np.diagflat(single_output) - \
                    np.dot(single_output, single_output.T)
            
            self.dinputs[index] = np.dot(jacobian_matrix , single_dvalues)

class Softmax_CCE_loss():
    def __init__(self):
        self.activation = Softmax()
        self.loss = categorical_cross_entropy()

    def forward(self , inputs , y_true):
        self.activation.forward(inputs)
        self.output = self.activation.ouptut

        return self.loss.calculate(self.output , y_true)
    
    def backward(self , dvalues , y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true =np.argmax(y_true , axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples