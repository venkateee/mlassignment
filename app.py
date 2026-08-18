"""
app.py  -  Streamlit front-end for the classification models.

Features (per assignment):
  a. Upload test CSV
  b. Model selection dropdown
  c. Evaluation metrics
  d. Confusion matrix + classification report
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

st.set_page_config(page_title="Classification Model Explorer", layout="wide")
st.title("Classification Model Explorer")
st.caption("Compare six classifiers on your uploaded test data.")

MODEL_DIR = "model"


@st.cache_resource
def load_models():
    models = {}
    for path in glob.glob(os.path.join(MODEL_DIR, "*.pkl")):
        name = os.path.basename(path).replace(".pkl", "")
        if name == "feature_names":
            continue
        pretty = name.replace("_", " ").title()
        models[pretty] = joblib.load(path)
    feats = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return models, feats


models, feature_cols = load_models()

# ---- 1. Data upload ------------------------------------------------------
st.sidebar.header("1. Test data")
uploaded = st.sidebar.file_uploader("Upload test CSV (features + 'target')", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif os.path.exists("test_data.csv"):
    df = pd.read_csv("test_data.csv")
    st.sidebar.info("Using bundled test_data.csv (upload your own to override).")
else:
    st.warning("Please upload a test CSV to begin.")
    st.stop()

if "target" not in df.columns:
    st.error("CSV must contain a 'target' column.")
    st.stop()

X_test = df[feature_cols]
y_test = df["target"]
n_classes = y_test.nunique()

st.write(f"**Test set:** {X_test.shape[0]} rows | {X_test.shape[1]} features | {n_classes} classes")
with st.expander("Preview data"):
    st.dataframe(df.head())

# ---- 2. Model selection --------------------------------------------------
st.sidebar.header("2. Model")
choice = st.sidebar.selectbox("Select a model", sorted(models.keys()))
model = models[choice]

# ---- 3. Predict + metrics ------------------------------------------------
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)
avg = "binary" if n_classes == 2 else "weighted"
auc = (roc_auc_score(y_test, proba[:, 1]) if n_classes == 2
       else roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))

metrics = {
    "Accuracy":  accuracy_score(y_test, y_pred),
    "AUC":       auc,
    "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
    "Recall":    recall_score(y_test, y_pred, average=avg, zero_division=0),
    "F1":        f1_score(y_test, y_pred, average=avg, zero_division=0),
    "MCC":       matthews_corrcoef(y_test, y_pred),
}

st.subheader(f"Evaluation metrics — {choice}")
cols = st.columns(6)
for c, (k, v) in zip(cols, metrics.items()):
    c.metric(k, f"{v:.3f}")

# ---- 4. Confusion matrix + classification report -------------------------
left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

# ---- 5. All-model comparison table ---------------------------------------
st.subheader("All models — comparison")
rows = []
for name, mdl in models.items():
    yp = mdl.predict(X_test)
    pr = mdl.predict_proba(X_test)
    a = (roc_auc_score(y_test, pr[:, 1]) if n_classes == 2
         else roc_auc_score(y_test, pr, multi_class="ovr", average="macro"))
    rows.append({
        "Model": name,
        "Accuracy":  accuracy_score(y_test, yp),
        "AUC":       a,
        "Precision": precision_score(y_test, yp, average=avg, zero_division=0),
        "Recall":    recall_score(y_test, yp, average=avg, zero_division=0),
        "F1":        f1_score(y_test, yp, average=avg, zero_division=0),
        "MCC":       matthews_corrcoef(y_test, yp),
    })
st.dataframe(pd.DataFrame(rows).set_index("Model").round(4))
