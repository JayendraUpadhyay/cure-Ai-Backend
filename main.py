from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import cv2
import pickle
import onnxruntime as ort
import io

app = FastAPI(title="Cure AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models ──────────────────────────────────────────────────
try:
    brain_session = ort.InferenceSession("brain_tumor_model.onnx")
    brain_input   = brain_session.get_inputs()[0].name

    diabetes_model = pickle.load(open("diabetes_model.pkl",  "rb"))
    heart_model    = pickle.load(open("heart_model.pkl",     "rb"))
    model_columns  = pickle.load(open("model_columns.pkl",   "rb"))
    scaler         = pickle.load(open("scaler.pkl",          "rb"))
    num_cols       = pickle.load(open("num_cols.pkl",        "rb"))
    print("✅ All models loaded successfully")
except Exception as e:
    print(f"⚠️  Model loading error: {e}")

# ── Schemas ───────────────────────────────────────────────────────
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

class HeartInput(BaseModel):
    age: float; sex: float; cp: float; trestbps: float
    chol: float; fbs: float; restecg: float; thalach: float
    exang: float; oldpeak: float; slope: float; ca: float; thal: float

# ── Helpers ───────────────────────────────────────────────────────
def preprocess_diabetes(d: DiabetesInput):
    df = pd.DataFrame([d.dict()])
    df["NewBMI"] = "Healthy"
    df.loc[df["BMI"] < 18.5, "NewBMI"] = "Underweight"
    df.loc[(df["BMI"] >= 25) & (df["BMI"] < 30), "NewBMI"] = "Overweight"
    df.loc[df["BMI"] >= 30, "NewBMI"] = "Obese"
    df["NewInsulinScore"] = np.where(
        (df["Insulin"] >= 16) & (df["Insulin"] <= 166), "Normal", "Abnormal")
    df["NewGlucose"] = "Normal"
    df.loc[df["Glucose"] > 126, "NewGlucose"] = "High"
    df = pd.get_dummies(df).reindex(columns=model_columns, fill_value=0)
    df[num_cols] = scaler.transform(df[num_cols])
    return df

# ── Routes ────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Cure AI API is running 🏥"}

@app.post("/predict/brain")
async def predict_brain(file: UploadFile = File(...)):
    contents = await file.read()
    arr  = np.frombuffer(contents, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")
    img  = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img  = np.expand_dims(img, 0)
    pred = brain_session.run(None, {brain_input: img})[0]
    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    probs = {c: float(pred[0][i]) for i, c in enumerate(classes)}
    result = classes[int(np.argmax(pred))]
    return {"result": result, "confidence": float(np.max(pred)),
            "probabilities": probs, "is_safe": result == "No Tumor"}

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    df   = preprocess_diabetes(data)
    pred = int(diabetes_model.predict(df)[0])
    prob = float(diabetes_model.predict_proba(df)[0][1])
    return {"result": "High Risk" if pred == 1 else "Low Risk",
            "probability": prob, "is_safe": pred == 0}

@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    row  = [[data.age, data.sex, data.cp, data.trestbps, data.chol,
             data.fbs, data.restecg, data.thalach, data.exang,
             data.oldpeak, data.slope, data.ca, data.thal]]
    pred = int(heart_model.predict(row)[0])
    try:
        prob = float(heart_model.predict_proba(row)[0][1])
    except:
        prob = 0.8 if pred == 1 else 0.2
    return {"result": "High Risk" if pred == 1 else "Low Risk",
            "probability": prob, "is_safe": pred == 0}
