import numpy as np

class ReLU_Activation:
    def forward(self , inputs):
        self.output = np.maximum(0 , inputs)

class Softmax_Activation:
    def forward(self , inputs):
        exp_value = np.exp(inputs - np.max(inputs , axis=1 , keepdims=True))

        probabilities = exp_value / np.sum(exp_value , axis=1 , keepdims=True)
        self.output = probabilities