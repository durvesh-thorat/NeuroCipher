import numpy as np
from network import CNN

model = CNN(num_filters1=32, num_filters2=64, filter_size=3, pool_size=2, hidden_size=512)
try:
    model.load(r"model/weights/cnn/cnn_adam_aug_99.36_lr0.00001_ep15.npz")
except FileNotFoundError:
    raise RuntimeError("CNN weights not found at model/weights/cnn/. Copy weights file first.")

def predict(X):
    output = model.forward(X)
    prediction = np.argmax(output)
    confidence = np.max(output)
    print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
    return prediction, confidence