import base64
import numpy as np
from PIL import Image, ImageOps
import io

def _process_pil(img):
    """Shared preprocessing logic — takes a PIL image, returns (1,1,28,28) numpy array."""
    img = img.convert("L")
    img = ImageOps.invert(img)              # white bg + black stroke → MNIST style
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img.thumbnail((24, 24), Image.LANCZOS)
    padded = Image.new("L", (28, 28), 0)
    offset = ((28 - img.width) // 2, (28 - img.height) // 2)
    padded.paste(img, offset)

    sample = np.array(padded, dtype=np.float32) / 255.0
    return sample.reshape(1, 1, 28, 28)

def preprocess(image_base64):
    """Web route — base64 PNG → (1,1,28,28) numpy array."""
    img_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_bytes))
    return _process_pil(img)

def preprocess_pil(img):
    """Tkinter route — PIL image → (1,1,28,28) numpy array."""
    return _process_pil(img)
