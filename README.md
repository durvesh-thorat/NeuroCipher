
# NeuroCipher

**Handwritten digit recognizer built entirely from scratch -- no ML frameworks, no PyTorch, no TensorFlow.**

A convolutional neural network implemented in pure NumPy, trained on MNIST to **99.36% accuracy**, with a FastAPI web interface and a Tkinter desktop GUI.

---

## Why this exists

Most people learning deep learning use `model.fit()` and call it a day. This project goes the other way -- every forward pass, every backward pass, every gradient update is written by hand. Backpropagation through conv layers, im2col vectorization, Adam optimizer, data augmentation -- all from first principles.

---

## Architecture

```
Input (1x28x28)
    |
    v
ConvLayer  -- 32 filters, 3x3 kernel, He init  -->  (32x26x26)
ReLU
    |
    v
ConvLayer  -- 64 filters, 3x3 kernel, He init  -->  (64x24x24)
ReLU
    |
    v
MaxPool    -- 2x2 window                        -->  (64x12x12)
Flatten                                         -->  (9216,)
    |
    v
DenseLayer -- 9216 -> 512, ReLU
DenseLayer -- 512  -> 10,  Softmax
    |
    v
Output (10 classes)
```

**Total trainable parameters: ~4.74 million**

| Layer | Parameters |
|---|---|
| ConvLayer 1 (32 filters, 3x3) | 320 |
| ConvLayer 2 (64 filters, 3x3) | 18,496 |
| DenseLayer 1 (9216 -> 512) | 4,719,104 |
| DenseLayer 2 (512 -> 10) | 5,130 |
| **Total** | **4,743,050** |

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | **99.36%** |
| Test set size | 10,000 images |
| Training set | 60,000 MNIST images + augmentation |
| Final optimizer | Adam (b1=0.9, b2=0.999, e=1e-8) |

### Training pipeline

| Stage | Optimizer | LR | Epochs | Accuracy |
|---|---|---|---|---|
| Scratch | SGD | 0.05 -> 0.001 | 55 | ~98.5% |
| Fine-tune 1 | Adam | 3e-4 | 15 | 99.21% |
| Fine-tune 2 | Adam | 3e-5 | 15 | 99.31% |
| Fine-tune 3 | Adam | 1e-5 | 15 | **99.36%** |

---

## What's implemented from scratch

- **ConvLayer** -- im2col via `np.lib.stride_tricks.as_strided`, vectorized forward and backward pass
- **MaxPoolLayer** -- fully vectorized forward and backward using reshape trick, no Python loops
- **FlattenLayer** -- shape-preserving reshape for backward pass
- **DenseLayer** -- matrix multiply with He initialization
- **ReLU, Softmax** -- with numerically stable Softmax (max subtraction)
- **Categorical cross-entropy loss**
- **Adam optimizer** -- per-parameter moment estimates, bias correction
- **Data augmentation** -- random +-2 pixel translation using `np.roll`
- **Model save/load** -- weights serialized with `np.savez`

---

## Run locally

```bash
git clone https://github.com/durvesh-thorat/NeuroCipher
cd NeuroCipher
pip install -r requirements.txt
```

**Web interface (FastAPI)**
```bash
uvicorn app:app --reload
```

**Desktop GUI (Tkinter)**
```bash
python main.py
```

---

## Live demo

[**Live Demo**](https://huggingface.co/spaces/durvesh-thorat/NeuroCipher)

---

## File structure

```
NeuroCipher/
├── layers.py        -- DenseLayer, ConvLayer, MaxPoolLayer, FlattenLayer
├── activations.py   -- ReLU, Softmax
├── loss.py          -- CategoricalCrossEntropyLoss
├── network.py       -- CNN (forward, backward, Adam update, train, save, load)
├── preprocess.py    -- image normalization and reshaping
├── predict.py       -- model loading and inference
├── app.py           -- FastAPI server
├── main.py          -- Tkinter desktop GUI
├── Dockerfile       -- for Hugging Face Spaces deployment
└── templates/
    └── index.html   -- canvas-based drawing interface
```

---

## Stack

- Python 3.9+
- NumPy (all computation)
- Pillow (image preprocessing)
- FastAPI + Uvicorn (web server)
- Tkinter (desktop GUI)

No PyTorch. No TensorFlow. No Keras. No scikit-learn.

---

## Author

**Durvesh Thorat** -- B.Tech IT, Pillai College of Engineering, Panvel