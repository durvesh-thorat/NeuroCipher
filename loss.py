import numpy as np

class CategoricalCrossEntropyLoss:

    def forward(self, A, y):

        m = y.shape[0]
        loss = -np.sum(y * np.log(A + 1e-8)) / m
        return loss
    
    def backward(self, A, y):

        m = y.shape[0]
        single_batch_dA = -y/A #because dL/dA = -y * 1/A
        dA = single_batch_dA / m
        return dA
    
