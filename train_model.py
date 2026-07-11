import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
import pickle
import json


# ── 1. Load Dataset ──────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("Diabetes_012_health.csv")

# ── 2. Explore & Check ───────────────────────────────────────────────────────
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Class distribution:\n{df['Diabetes_012'].value_counts()}")

# ── 3. Preprocessing ─────────────────────────────────────────────────────────
# Separate features and target
X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

# Handle missing values (fill with median)
X = X.fillna(X.median())

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── 5. Train Random Forest Model ──────────────────────────────────────────────
print("\nTraining Random Forest model (this may take 1-2 minutes)...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1       # use all CPU cores
)
model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred_binary = (y_pred > 0).astype(int)
y_test_binary = (y_test > 0).astype(int)

metrics = {
    "accuracy":  round(accuracy_score(y_test_binary, y_pred_binary), 4),
    "precision": round(precision_score(y_test_binary, y_pred_binary), 4),
    "recall":    round(recall_score(y_test_binary, y_pred_binary), 4),
    "f1_score":  round(f1_score(y_test_binary, y_pred_binary), 4),
    "roc_auc":   round(roc_auc_score(y_test_binary, y_prob), 4),
}

print("\n📊 Model Evaluation Metrics:")
for k, v in metrics.items():
    print(f"  {k:<12}: {v}")

print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_binary, target_names=["Non-Diabetic", "Diabetic"]))


# ── 7. Save Model & Scaler ────────────────────────────────────────────────────
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "features": list(df.drop("Diabetes_012", axis=1).columns)}, f)

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("\n✅ Model saved as 'diabetes_model.pkl'")
print("✅ Metrics saved as 'metrics.json'")