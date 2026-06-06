import numpy as np
from layers import *
from activations import *
from loss import CategoricalCrossEntropyLoss

class CNN:
    def __init__(self, num_filters1, num_filters2, filter_size, pool_size, hidden_size, input_size=28):
        self.relu = ReLU()
        self.softmax = Softmax()

        self.C1 = ConvLayer(num_filters1, filter_size)
        self.C2 = ConvLayer(num_filters2, filter_size)
        self.MXP1 = MaxPoolLayer(pool_size)
        self.FL1 = FlattenLayer()

        # 28 -> conv1 -> 26 -> conv2 -> 24 -> pool -> 12
        conv1_out = input_size - filter_size + 1
        conv2_out = conv1_out - filter_size + 1
        pool_out = conv2_out // pool_size
        flatten_size = num_filters2 * pool_out ** 2

        self.DL1 = DenseLayer(flatten_size, hidden_size)
        self.DL2 = DenseLayer(hidden_size, 10)

        # Adam parameters
        self.t = 0
        self.m = {}
        self.v = {}
        for name, param in [('C1W', self.C1.W), ('C1b', self.C1.b),
                            ('C2W', self.C2.W), ('C2b', self.C2.b),
                            ('DL1W', self.DL1.W), ('DL1b', self.DL1.b),
                            ('DL2W', self.DL2.W), ('DL2b', self.DL2.b)]:
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

        self.loss = CategoricalCrossEntropyLoss()
        self.losses = []

    def forward(self, X):
        self.fm1 = self.C1.forward(X)
        self.afm1 = self.relu.forward(self.fm1)
        self.fm2 = self.C2.forward(self.afm1)
        self.afm2 = self.relu.forward(self.fm2)
        self.maxpooled = self.MXP1.forward(self.afm2)
        self.flattened = self.FL1.forward(self.maxpooled)
        self.Z1 = self.DL1.forward(self.flattened)
        self.A1 = self.relu.forward(self.Z1)
        self.Z2 = self.DL2.forward(self.A1)
        self.A2 = self.softmax.forward(self.Z2)
        return self.A2

    def backward(self, y):
        dZ2 = self.softmax.backward(A=self.A2, y=y)
        dA1 = self.DL2.backward(X=self.A1, dZ=dZ2)
        dZ1 = self.relu.backward(Z=self.Z1, dZ=dA1)
        d_flattened = self.DL1.backward(X=self.flattened, dZ=dZ1)
        d_maxpooled = self.FL1.backward(dX=d_flattened)
        d_afm2 = self.MXP1.backward(dMP=d_maxpooled)
        d_fm2 = self.relu.backward(Z=self.fm2, dZ=d_afm2)
        _, _, d_afm1 = self.C2.backward(dZ=d_fm2)
        d_fm1 = self.relu.backward(Z=self.fm1, dZ=d_afm1)
        self.dW, self.db, self.dX = self.C1.backward(dZ=d_fm1)

    def update(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        params = [
            ('C1W', self.C1, 'W', self.C1.dW),
            ('C1b', self.C1, 'b', self.C1.db),
            ('C2W', self.C2, 'W', self.C2.dW),
            ('C2b', self.C2, 'b', self.C2.db),
            ('DL1W', self.DL1, 'W', self.DL1.dW),
            ('DL1b', self.DL1, 'b', self.DL1.db),
            ('DL2W', self.DL2, 'W', self.DL2.dW),
            ('DL2b', self.DL2, 'b', self.DL2.db),
        ]
        for name, layer, attr, grad in params:
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * grad ** 2
            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)
            setattr(layer, attr, getattr(layer, attr) - lr * m_hat / (np.sqrt(v_hat) + eps))

    def save(self, path):
        np.savez(path,
            W1=np.asnumpy(self.DL1.W), W2=np.asnumpy(self.DL2.W),
            B1=np.asnumpy(self.DL1.b), B2=np.asnumpy(self.DL2.b),
            WC1=np.asnumpy(self.C1.W), BC1=np.asnumpy(self.C1.b),
            WC2=np.asnumpy(self.C2.W), BC2=np.asnumpy(self.C2.b))

    def load(self, path):
        data = __import__('numpy').load(path)
        self.DL1.W = np.array(data['W1'])
        self.DL2.W = np.array(data['W2'])
        self.DL1.b = np.array(data['B1'])
        self.DL2.b = np.array(data['B2'])
        self.C1.W = np.array(data['WC1'])
        self.C1.b = np.array(data['BC1'])
        self.C2.W = np.array(data['WC2'])
        self.C2.b = np.array(data['BC2'])
