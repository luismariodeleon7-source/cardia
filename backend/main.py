from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import json
import io
import os


BASE_DIR = os.path.dirname(__file__)

MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

MODEL_PATH_LR = os.path.join(MODELS_DIR, "logistic_regression.pkl")
MODEL_PATH_MLP = os.path.join(MODELS_DIR, "neural_network.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

app = FastAPI(
    title="Heart Disease Predictor API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)


try:
    lr = joblib.load(MODEL_PATH_LR)
    mlp = joblib.load(MODEL_PATH_MLP)
    scaler = joblib.load(SCALER_PATH)

    with open(METRICS_PATH, "r") as f:
        SAVED_METRICS = json.load(f)

    print("✅ Modelos cargados correctamente")

except Exception as e:
    raise RuntimeError(f"❌ Error cargando modelos: {e}")


FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "Diabetes",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income"
]


class PatientData(BaseModel):
    HighBP: float
    HighChol: float
    CholCheck: float
    BMI: float
    Smoker: float
    Stroke: float
    Diabetes: float
    PhysActivity: float
    Fruits: float
    Veggies: float
    HvyAlcoholConsump: float
    AnyHealthcare: float
    NoDocbcCost: float
    GenHlth: float
    MentHlth: float
    PhysHlth: float
    DiffWalk: float
    Sex: float
    Age: float
    Education: float
    Income: float


class PredictionResponse(BaseModel):
    model: str
    prediction: int
    probability: float
    risk_level: str
    message: str


def to_array(p: PatientData):
    return np.array([[getattr(p, f) for f in FEATURES]])


def risk_label(prob):
    if prob < 0.3:
        return "BAJO", "Riesgo bajo de enfermedad cardíaca."
    elif prob < 0.6:
        return "MODERADO", "Riesgo moderado. Consulte un médico."
    else:
        return "ALTO", "Riesgo alto. Consulte un cardiólogo."


def predict(model, patient, name):
    X = scaler.transform(to_array(patient))

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    risk, msg = risk_label(prob)

    return PredictionResponse(
        model=name,
        prediction=pred,
        probability=round(prob, 4),
        risk_level=risk,
        message=msg
    )


@app.get("/")
def home():
    return FileResponse(
        os.path.join(FRONTEND_DIR, "index.html")
    )


@app.get("/metrics")
def metrics():
    return SAVED_METRICS


@app.post("/predict/logistic", response_model=PredictionResponse)
def predict_lr(p: PatientData):
    return predict(lr, p, "Regresión Logística")


@app.post("/predict/neural", response_model=PredictionResponse)
def predict_mlp(p: PatientData):
    return predict(mlp, p, "Red Neuronal")


@app.post("/predict/both")
def predict_both(p: PatientData):
    return {
        "logistic_regression": predict(
            lr,
            p,
            "Regresión Logística"
        ),
        "neural_network": predict(
            mlp,
            p,
            "Red Neuronal"
        )
    }

async def batch_predict(file, model, name):

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Debe ser un archivo CSV"
        )

    df = pd.read_csv(
        io.BytesIO(await file.read())
    )

    
    X = scaler.transform(
        df[FEATURES].astype(float).values
    )

    preds = model.predict(X).tolist()
    probs = model.predict_proba(X)[:, 1].tolist()

    
    metrics = {}

    if "HeartDiseaseorAttack" in df.columns:

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix
        )

        y_true = df["HeartDiseaseorAttack"].astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds)),
            "recall": float(recall_score(y_true, preds)),
            "f1": float(f1_score(y_true, preds)),
            "confusion_matrix": confusion_matrix(y_true, preds).tolist()
        }

    return {
        "model": name,
        "total_samples": len(preds),
        "predictions": preds,
        "probabilities": [round(p, 4) for p in probs],
        "metrics": metrics
    }
    return {
        "model": name,
        "total_samples": len(preds),
        "predictions": preds,
        "probabilities": [round(p, 4) for p in probs],
        "metrics": metrics
    }


@app.post("/predict/batch/logistic")
async def batch_lr(
    file: UploadFile = File(...)
):
    return await batch_predict(
        file,
        lr,
        "Regresión Logística"
    )


@app.post("/predict/batch/neural")
async def batch_mlp(
    file: UploadFile = File(...)
):
    return await batch_predict(
        file,
        mlp,
        "Red Neuronal"
    )