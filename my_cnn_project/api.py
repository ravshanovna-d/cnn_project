from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI(title="Malaria Detection API")

model = load_model("malaria_cnn_model.h5", compile=False)

IMG_SIZE = 128  

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)
        return image_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload a valid image")

    try:
        contents = await file.read()
        image = preprocess_image(contents)
        prediction = model.predict(image)[0][0]

        label = "Uninfected" if prediction > 0.5 else "Parasitized"
        confidence = float(prediction if label == "Uninfected" else 1 - prediction)

        return JSONResponse({
            "prediction": label,
            "confidence": round(confidence, 4),
            "raw_output": round(float(prediction), 4),
            "status": "success"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Malaria Detection API — Upload an image at /predict"}
