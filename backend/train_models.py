import pandas as pd, json, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


df = pd.read_csv("backend/heart.csv", na_values="?", low_memory=False)
df = df.dropna()


df.columns = [
    "HeartDiseaseorAttack","HighBP","HighChol","CholCheck","BMI","Smoker",
    "Stroke","Diabetes","PhysActivity","Fruits","Veggies",
    "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth",
    "MentHlth","PhysHlth","DiffWalk","Sex","Age",
    "Education","Income"
]


df["target"] = df["HeartDiseaseorAttack"]


FEATURES = [
    "HighBP","HighChol","CholCheck","BMI","Smoker",
    "Stroke","Diabetes","PhysActivity","Fruits","Veggies",
    "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth",
    "MentHlth","PhysHlth","DiffWalk","Sex","Age",
    "Education","Income"
]

X = df[FEATURES].values
y = df["target"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)


print("Entrenando Regresión Logística...")
lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
lr.fit(Xtr, y_train)

lp = lr.predict(Xte)
lpr = lr.predict_proba(Xte)[:, 1]

print(classification_report(y_test, lp))


print("Entrenando MLP (64-32)...")

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)

mlp.fit(Xtr, y_train)

mp = mlp.predict(Xte)
mpr = mlp.predict_proba(Xte)[:, 1]

print(classification_report(y_test, mp))

os.makedirs("models", exist_ok=True)

joblib.dump(lr, "models/logistic_regression.pkl")
joblib.dump(mlp, "models/neural_network.pkl")
joblib.dump(scaler, "models/scaler.pkl")


def md(yt, yp, yprob):
    return {
        "accuracy": round(float(accuracy_score(yt, yp)), 4),
        "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
        "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
        "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(yt, yprob)), 4),
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
    }

metrics = {
    "logistic_regression": md(y_test, lp, lpr),
    "neural_network": md(y_test, mp, mpr),
    "features": FEATURES,
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Entrenamiento completado correctamente")