from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from predict import predict
from pydantic import BaseModel
from preprocess import preprocess

class ImageInput(BaseModel):
    image: str

app = FastAPI(title="NeuroCipher")

@app.get("/")
def canvas():
    with open("templates/index.html", "r") as file:
        content = file.read()
        if content:
            return HTMLResponse(content=content, status_code=200)

@app.post("/predict")
def predict_digit(data: ImageInput):

    try:
        processed = preprocess(data.image)
    except Exception:
        return {
            "error": "Invalid image data"
            }

    output = predict(processed)
    return {
    "Predicted Digit": int(output[0]),
    "Confidence": float(output[1])
    }
