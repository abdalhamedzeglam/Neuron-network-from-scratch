import numpy as np

class Loss:
    def calculate(self , output , y):
        samples_losses = self.forward(output , y)
        data_loss = np.mean(samples_losses)

        return data_loss

class categorical_cross_entropy(Loss):

    def forward(self , y_pred , y_true):
        samples = len(y_true)

        y_pred_clipped = np.clip(y_pred , 1e-7 , 1- 1e-7)

        if len(y_true.shape) == 1:
            currect_confidences = y_pred_clipped[range(samples) , y_true]

        elif len(y_true.shape) == 2:
            currect_confidences = np.sum(y_pred_clipped * y_true , axis=1)

        negative_log = -np.log(currect_confidences)
        return negative_log
    
    def backward(self , dvalues , y_true):

        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs /= samples