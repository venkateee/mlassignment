"""
train_models.py
Trains classification models on a chosen dataset, prints an evaluation-metrics
comparison table, and saves each trained pipeline + the test data (CSV).

Run once locally / on BITS Virtual Lab:
    python train_models.py

Outputs:
    model/<model>.pkl        one saved pipeline per model
    model/feature_names.pkl  column order the app expects
    test_data.csv            held-out test set (features + 'target' column)
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits          # <-- swap this to change dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC                        # optional 6th model

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

RANDOM_STATE = 42
os.makedirs("model", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
#    To use YOUR OWN csv instead of the built-in dataset, comment the block
#    below and do:  df = pd.read_csv("your_file.csv");  target col named 'target'
# ---------------------------------------------------------------------------
data = load_digits(as_frame=True)
df = data.frame.rename(columns={"target": "target"})
feature_cols = [c for c in df.columns if c != "target"]

X = df[feature_cols]
y = df["target"]
n_classes = y.nunique()
print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features, {n_classes} classes")

# ---------------------------------------------------------------------------
# 2. Train / test split  (we ship the TEST set as test_data.csv)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Define models. Every model is wrapped in a StandardScaler pipeline so the
#    Streamlit app can just call predict() on raw features.
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression":     make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Decision Tree":           make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=RANDOM_STATE)),
    "kNN":                     make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes":             make_pipeline(StandardScaler(), GaussianNB()),
    "Random Forest":           make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    "SVM":                     make_pipeline(StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE)),  # optional
}


def evaluate(model, X_te, y_te):
    """Return the six required metrics for one fitted model."""
    y_pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    avg = "binary" if n_classes == 2 else "weighted"

    if n_classes == 2:
        auc = roc_auc_score(y_te, proba[:, 1])
    else:
        auc = roc_auc_score(y_te, proba, multi_class="ovr", average="macro")

    return {
        "Accuracy":  accuracy_score(y_te, y_pred),
        "AUC":       auc,
        "Precision": precision_score(y_te, y_pred, average=avg, zero_division=0),
        "Recall":    recall_score(y_te, y_pred, average=avg, zero_division=0),
        "F1":        f1_score(y_te, y_pred, average=avg, zero_division=0),
        "MCC":       matthews_corrcoef(y_te, y_pred),
    }


rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    rows.append({"Model": name, **metrics})
    fname = f"model/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, fname)
    print(f"saved {fname}")

# save the column order so the app can validate uploads
joblib.dump(feature_cols, "model/feature_names.pkl")

# ---------------------------------------------------------------------------
# 4. Comparison table + test data
# ---------------------------------------------------------------------------
results = pd.DataFrame(rows).set_index("Model").round(4)
print("\n===== Comparison Table =====")
print(results.to_string())
results.to_csv("model/metrics_summary.csv")

test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"\nsaved test_data.csv ({test_df.shape[0]} rows)")
