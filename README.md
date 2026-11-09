<div align="center">

# 🦠 Malaria Detection with CNN

**A convolutional neural network that classifies blood-cell images as *Parasitized* or *Uninfected*, served through a FastAPI endpoint.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-CNN-D00000?logo=keras&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20accuracy-94%25-brightgreen)

</div>

---

## 📌 Overview

Malaria is diagnosed by looking at blood smears under a microscope, which is slow manual work. This project trains a **CNN from scratch** on the TensorFlow [`malaria`](https://www.tensorflow.org/datasets/catalog/malaria) dataset (27,558 cell images). It then exposes the model through a **REST API**: you upload an image and get a prediction back.

## 📊 Results

Evaluated on a held-out 20% split (5,512 images):

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Uninfected | 0.94 | 0.93 | 0.94 |
| Parasitized | 0.93 | 0.94 | 0.93 |
| **Accuracy** | | | **0.94** |

## 🧠 Model

```
Input (128×128×3)
 → Conv2D(32) + MaxPool
 → Conv2D(64) + MaxPool
 → Conv2D(128) + MaxPool
 → Flatten → Dense(64) → Dropout(0.2)
 → Dense(1, sigmoid)
```

- **Preprocessing:** resize to 128×128, scale to [0, 1]
- **Augmentation:** random horizontal flip
- **Training:** Adam (lr = 5e-6), binary cross-entropy, up to 60 epochs with EarlyStopping (patience 10)

## 🔌 API

`my_cnn_project/api.py` loads `malaria_cnn_model.h5` and serves predictions.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health message |
| `POST` | `/predict/` | Upload a cell image and get the class and confidence |

**Example response**

```json
{
  "prediction": "Parasitized",
  "confidence": 0.9731,
  "raw_output": 0.0269,
  "status": "success"
}
```

## 🚀 Getting Started

```bash
git clone https://github.com/dlsoul13/cnn_project.git
cd cnn_project/my_cnn_project

pip install fastapi uvicorn tensorflow pillow numpy python-multipart
uvicorn api:app --reload
```

Open **http://127.0.0.1:8000/docs** to try `/predict/` from the Swagger UI, or use curl:

```bash
curl -X POST -F "file=@cell.png" http://127.0.0.1:8000/predict/
```

## 📁 Project Structure

```
cnn_project/
└── my_cnn_project/
    ├── Untitled (1).ipynb     # data loading, training, evaluation (Colab)
    ├── api.py                 # FastAPI inference service
    └── malaria_cnn_model.h5   # trained model
```

## 🗺 Roadmap

- [ ] Transfer learning (EfficientNet / MobileNet) for better accuracy
- [ ] Grad-CAM visualizations of infected regions
- [ ] Docker image and simple web UI for uploads

> ⚠️ This is an educational project, not a medical device.
