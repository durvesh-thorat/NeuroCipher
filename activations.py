import cupy as np

class ReLU:
    def forward(self, Z):
        return np.maximum(0, Z)
    
    def backward(self, Z, dZ):
        return dZ * (Z > 0)
    
class Softmax:
    def forward(self, Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        return A
    
    def backward(self, A, y):
        
        m = y.shape[0]
        # dL/dZ = A - y (combined Softmax + CrossEntropy gradient)
        return (A-y) / m #dZ
