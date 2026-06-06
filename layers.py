import numpy as np
from numpy.lib.stride_tricks import as_strided

def get_patches(X, filter_size):
    batch, channels, h, w = X.shape
    out_h = h - filter_size + 1
    out_w = w - filter_size + 1

    shape = (batch, out_h, out_w, filter_size, filter_size)
    strides = (X.strides[0], X.strides[2], X.strides[3], X.strides[2], X.strides[3])

    patches = as_strided(X[:, 0, :, :], shape=shape, strides=strides).copy()  # .copy() — CuPy non-contiguous fix
    return patches

class DenseLayer:
    def __init__(self, input_size, output_size):
        std_dev = np.sqrt(2 / input_size)
        self.W = np.random.randn(input_size, output_size) * std_dev
        self.b = np.zeros((1, output_size))

    def forward(self, X):
        Z = np.dot(X, self.W) + self.b
        return Z

    def backward(self, X, dZ):
        m = X.shape[0]
        self.dW = np.dot(X.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = np.dot(dZ, self.W.T)
        return dX

class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.filter_size = filter_size
        self.W = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2 / filter_size**2)
        self.b = np.zeros(num_filters)

    def forward(self, X):
        batch, channels, h, w = X.shape
        out_h = h - self.filter_size + 1
        out_w = w - self.filter_size + 1

        patches = get_patches(X, self.filter_size)
        patches_flat = patches.reshape(batch, out_h * out_w, -1)
        W_flat = self.W.reshape(self.W.shape[0], -1)

        Z = np.dot(patches_flat, W_flat.T) + self.b
        self.Z = Z.reshape(batch, out_h, out_w, self.W.shape[0]).transpose(0, 3, 1, 2)

        self.X = X
        self.patches_flat = patches_flat
        self.out_h = out_h
        self.out_w = out_w
        return self.Z

    def backward(self, dZ):
        batch = self.X.shape[0]

        dZ_reshaped = dZ.transpose(0, 2, 3, 1).reshape(-1, self.W.shape[0])
        patches_reshaped = self.patches_flat.reshape(-1, self.filter_size * self.filter_size)

        dW_flat = np.dot(dZ_reshaped.T, patches_reshaped)
        self.dW = dW_flat.reshape(self.W.shape)
        self.db = np.sum(dZ, axis=(0, 2, 3))

        W_flat = self.W.reshape(self.W.shape[0], -1)
        dpatches_flat = np.dot(dZ_reshaped, W_flat)
        dpatches = dpatches_flat.reshape(batch, self.out_h, self.out_w, self.filter_size, self.filter_size)

        dX = np.zeros(self.X.shape)
        for i in range(self.filter_size):
            for j in range(self.filter_size):
                dX[:, 0, i:i+self.out_h, j:j+self.out_w] += dpatches[:, :, :, i, j]

        return self.dW, self.db, dX

class MaxPoolLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, Z):
        batch, filters, h, w = Z.shape
        self.MP_height = h // self.pool_size
        self.MP_width = w // self.pool_size

        Z_reshaped = Z.reshape(batch, filters, self.MP_height, self.pool_size, self.MP_width, self.pool_size)
        MP = Z_reshaped.max(axis=(3, 5))

        self.Z = Z
        self.MP = MP
        return MP

    def backward(self, dMP):
        batch, filters, h, w = self.Z.shape
        Z_reshaped = self.Z.reshape(batch, filters, self.MP_height, self.pool_size, self.MP_width, self.pool_size)

        max_vals = self.MP[:, :, :, np.newaxis, :, np.newaxis]
        mask = (Z_reshaped == max_vals)

        dMP_expanded = dMP[:, :, :, np.newaxis, :, np.newaxis] * mask
        dZ = dMP_expanded.reshape(batch, filters, h, w)
        return dZ

class FlattenLayer:
    def forward(self, X):
        self.input_shape = X.shape
        batch_size = X.shape[0]
        flattened_size = X.shape[1] * X.shape[2] * X.shape[3]
        return X.reshape(batch_size, flattened_size)

    def backward(self, dX):
        return dX.reshape(self.input_shape)
