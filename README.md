---
title: NeuroCipher
emoji: "1F522"
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---
# NeuroCipher

**Handwritten digit recognizer built entirely from scratch â€” no ML frameworks, no PyTorch, no TensorFlow.**

A convolutional neural network implemented in pure NumPy, trained on MNIST to **99.36% accuracy**, with a FastAPI web interface and a Tkinter desktop GUI.

---

## Why this exists

Most people learning deep learning use `model.fit()` and call it a day. This project goes the other way â€” every forward pass, every backward pass, every gradient update is written by hand. Backpropagation through conv layers, im2col vectorization, Adam optimizer, data augmentation â€” all from first principles.

---

## Architecture

```
Input (1Ã—28Ã—28)
    â”‚
    â–¼
ConvLayer  â€” 32 filters, 3Ã—3 kernel, He init  â†’  (32Ã—26Ã—26)
ReLU
    â”‚
    â–¼
ConvLayer  â€” 64 filters, 3Ã—3 kernel, He init  â†’  (64Ã—24Ã—24)
ReLU
    â”‚
    â–¼
MaxPool    â€” 2Ã—2 window                        â†’  (64Ã—12Ã—12)
Flatten                                        â†’  (9216,)
    â”‚
    â–¼
DenseLayer â€” 9216 â†’ 512, ReLU
DenseLayer â€” 512  â†’ 10,  Softmax
    â”‚
    â–¼
Output (10 classes)
```

**Total trainable parameters: ~4.74 million**

| Layer | Parameters |
|---|---|
| ConvLayer 1 (32 filters, 3Ã—3) | 320 |
| ConvLayer 2 (64 filters, 3Ã—3) | 18,496 |
| DenseLayer 1 (9216 â†’ 512) | 4,719,104 |
| DenseLayer 2 (512 â†’ 10) | 5,130 |
| **Total** | **4,743,050** |

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | **99.36%** |
| Test set size | 10,000 images |
| Training set | 60,000 MNIST images + augmentation |
| Final optimizer | Adam (Î²â‚=0.9, Î²â‚‚=0.999, Îµ=1e-8) |

### Training pipeline

| Stage | Optimizer | LR | Epochs | Accuracy |
|---|---|---|---|---|
| Scratch | SGD | 0.05 â†’ 0.001 | 55 | ~98.5% |
| Fine-tune 1 | Adam | 3e-4 | 15 | 99.21% |
| Fine-tune 2 | Adam | 3e-5 | 15 | 99.31% |
| Fine-tune 3 | Adam | 1e-5 | 15 | **99.36%** â† best |

---

## What's implemented from scratch

- **ConvLayer** â€” im2col via `np.lib.stride_tricks.as_strided`, vectorized forward and backward pass
- **MaxPoolLayer** â€” fully vectorized forward and backward using reshape trick, no Python loops
- **FlattenLayer** â€” shape-preserving reshape for backward pass
- **DenseLayer** â€” matrix multiply with He initialization
- **ReLU, Softmax** â€” with numerically stable Softmax (max subtraction)
- **Categorical cross-entropy loss**
- **Adam optimizer** â€” per-parameter moment estimates, Î²â‚/Î²â‚‚ bias correction
- **Data augmentation** â€” random Â±2 pixel translation using `np.roll`
- **Model save/load** â€” weights serialized with `np.savez`

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
# â†’ http://localhost:8000
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
â”œâ”€â”€ layers.py        â€” DenseLayer, ConvLayer, MaxPoolLayer, FlattenLayer
â”œâ”€â”€ activations.py   â€” ReLU, Softmax
â”œâ”€â”€ loss.py          â€” CategoricalCrossEntropyLoss
â”œâ”€â”€ network.py       â€” CNN (forward, backward, Adam update, train, save, load)
â”œâ”€â”€ preprocess.py    â€” image normalization and reshaping (28Ã—28 â†’ 1Ã—1Ã—28Ã—28)
â”œâ”€â”€ predict.py       â€” model loading and inference
â”œâ”€â”€ app.py           â€” FastAPI server
â”œâ”€â”€ main.py          â€” Tkinter desktop GUI
â”œâ”€â”€ Dockerfile       â€” for Hugging Face Spaces deployment
â””â”€â”€ templates/
    â””â”€â”€ index.html   â€” canvas-based drawing interface
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

**Durvesh Thorat** â€” B.Tech IT, Pillai College of Engineering, Panvel

