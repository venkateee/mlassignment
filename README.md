# ML Assignment 2 — Classification Model Explorer

## a. Problem statement
<!-- 2-3 sentences IN YOUR OWN WORDS. e.g.: "This project trains and compares
several classification algorithms on the <dataset> dataset and exposes them
through an interactive Streamlit app so a user can upload test data, pick a
model, and inspect its evaluation metrics and confusion matrix." -->

## b. Dataset description
<!-- Name, source (UCI/Kaggle link), what each instance represents,
number of instances, number of features, number of classes, and what
'target' means. Fill in YOUR dataset here. -->
- Source:
- Instances:
- Features:
- Classes:
- Target variable:

## c. GitHub Repository Link
<!-- Paste your repo URL -->
https://github.com/<your-username>/<your-repo>

## d. Models used

### Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9778 | 0.9991 | 0.9780 | 0.9778 | 0.9778 | 0.9753 |
| Decision Tree | 0.8267 | 0.9032 | 0.8277 | 0.8267 | 0.8263 | 0.8076 |
| kNN | 0.9644 | 0.9922 | 0.9651 | 0.9644 | 0.9643 | 0.9606 |
| Naive Bayes | 0.7644 | 0.9613 | 0.8223 | 0.7644 | 0.7624 | 0.7452 |
| Random Forest | 0.9667 | 0.9990 | 0.9671 | 0.9667 | 0.9665 | 0.9630 |
| SVM (optional 6th) | 0.9800 | 0.9992 | 0.9803 | 0.9800 | 0.9800 | 0.9778 |

> Numbers above are from the shipped `digits` scaffold. **Re-run `train_models.py`
> and paste YOUR numbers** — especially if you swap the dataset.

### Observations (write these yourself — 3 marks)

| ML Model | Observation about model performance |
|---|---|
| Logistic Regression | <!-- e.g. strong linear baseline; scaled features help --> |
| Decision Tree | <!-- e.g. lowest of the tree-family here; overfits single tree --> |
| kNN | <!-- e.g. competitive once features are standardized --> |
| Naive Bayes | <!-- e.g. weakest; feature-independence assumption violated --> |
| Random Forest | <!-- e.g. robust, near-top without tuning --> |
| Overall winner | <!-- name the model + one line on why --> |

## How to run locally
```bash
pip install -r requirements.txt
python train_models.py     # trains models, writes model/*.pkl and test_data.csv
streamlit run app.py       # launches the web app
```

## Live app
<!-- Paste your Streamlit Community Cloud URL -->
https://<your-app>.streamlit.app
