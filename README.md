# ML Assignment 2 — Classification Model Explorer

## a. Problem statement
This project implements and compares five supervised classification algorithms
on a single multi-class dataset and exposes them through an interactive Streamlit
web application. The app lets a user upload a test CSV, select any trained model,
and immediately view its evaluation metrics, confusion matrix, and classification
report — demonstrating a complete machine learning workflow from model training
through to deployment.

## b. Dataset description
- **Name:** Optical Recognition of Handwritten Digits
- **Source:** UCI Machine Learning Repository (also bundled in scikit-learn as `load_digits`)
- **Instances:** 1,797
- **Features:** 64 — each feature is the grayscale intensity (0–16) of one pixel
  in an 8×8 image of a handwritten digit
- **Classes:** 10 (digits 0–9) — a multi-class classification problem
- **Target:** the digit label (0–9)
- **Split:** 75% train / 25% test (450 test rows shipped as `test_data.csv`)
- **Preprocessing:** all features standardized with StandardScaler before training

## c. GitHub Repository Link
https://github.com/venkateee/mlassignment

## d. Models used

### Comparison Table
AUC is computed as a one-vs-rest macro average; precision, recall, and F1 use
weighted averages (multi-class dataset).

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9778 | 0.9991 | 0.9780 | 0.9778 | 0.9778 | 0.9753 |
| Decision Tree | 0.8267 | 0.9032 | 0.8277 | 0.8267 | 0.8263 | 0.8076 |
| kNN | 0.9644 | 0.9922 | 0.9651 | 0.9644 | 0.9643 | 0.9606 |
| Naive Bayes | 0.7644 | 0.9613 | 0.8223 | 0.7644 | 0.7624 | 0.7452 |
| Random Forest | 0.9667 | 0.9990 | 0.9671 | 0.9667 | 0.9665 | 0.9630 |

### Observations

| ML Model | Observation about model performance |
|---|---|
| Logistic Regression | Strong (0.978); the scaled pixel features are largely linearly separable across digit classes. |
| Decision Tree | Weakest here (0.827); a single tree overfits its training split and generalises poorly. |
| kNN | Competitive (0.964); distance matching works well once features are standardised, as similar digits cluster together. |
| Naive Bayes | Lowest overall (0.764); its assumption that pixels are independent is strongly violated, since neighbouring pixels correlate. |
| Random Forest | Robust and near the top (0.967) with no tuning; averaging many trees corrects the single tree's overfitting. |
| Overall winner | Logistic Regression (0.9778 accuracy, 0.9991 AUC), the strongest of the five required models; the standardized pixel features are largely linearly separable across the ten classes, edging out the tree-based and distance-based methods. |

## How to run locally

pip install -r requirements.txt
python train_models.py
streamlit run app.py

## Live app
https://mlassignment-txvopmmny8uj6mxa3axivd.streamlit.app/
