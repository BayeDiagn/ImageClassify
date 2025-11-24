from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
from fastapi.responses import JSONResponse
import tensorflow as tf

import logging
from datetime import datetime
import json

from fastapi.middleware.cors import CORSMiddleware
from config import CORS_CONFIG







# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les modèles
model_cnn = load_model("models/model_cnn.h5")
model_mobileNetV2 = load_model("models/model_MobileNetV2.h5")
model_googleNet = load_model("models/model_googlenet.h5")

# Charger les temps d'entraînement
with open("models/training_time_cnn.json") as f:
    training_time_cnn = json.load(f)["training_time_cnn"]
    print(training_time_cnn)
    
with open("models/training_time_googlenet.json") as f:
    training_time_googleNet = json.load(f)["training_time_googlenet"]

with open("models/training_time_MobileNetV2.json") as f:
    training_time_mobileNetV2 = json.load(f)["training_time_MobileNetV2"]

training_times = {
    "CNN": training_time_cnn,
    "MobileNetV2": training_time_mobileNetV2,
    "GoogleNet": training_time_googleNet
}

# Charger la liste des noms de classes
with open("models/nom_classes.json", "r") as f:
    nom_classes = json.load(f)


# Initialiser FastAPI
app = FastAPI(
    title="API Classification Caltech101",
    description="API pour classifier des images avec Transfer Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_CONFIG["allow_origins"],
    allow_credentials=CORS_CONFIG["allow_credentials"],
    allow_methods=CORS_CONFIG["allow_methods"],
    allow_headers=CORS_CONFIG["allow_headers"],
)


# Prétraitement de l'image
def preprocess_image(image: Image.Image, img_size=(224, 224)):
    image = image.resize(img_size)
    image_array = img_to_array(image) / 255.0
    image_array = tf.expand_dims(image_array, axis=0)
    return image_array


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint pour prédire la classe d'une image
    """
    # Lire l'image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Prétraiter
    image = image.resize((224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = tf.expand_dims(image_array, axis=0)
    
    # Prétraiter
    image_array = preprocess_image(image)

    # Prédictions
    def get_pred(model, img_array):
        preds = model.predict(img_array)
        classe_id = int(np.argmax(preds[0]))
        probabilite = float(preds[0][classe_id])
        return {"classe_id": classe_id, "nom_classe": nom_classes[classe_id], "probabilite": probabilite}


    predictions = {
        "cnn": {"model_name": "CNN", **get_pred(model_cnn, image_array)},
        "googleNet": {"model_name": "GoogleNet", **get_pred(model_googleNet, image_array)},
        "mobileNetv2": {"model_name": "MobileNetV2", **get_pred(model_mobileNetV2, image_array)}
    }
    
    return {"training_times": training_times, "predictions": predictions}