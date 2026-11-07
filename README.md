# Machine Learning Assignment 2

## Problem Statement

Develop and compare five machine learning classification models using the Bank Marketing Dataset and deploy the best-performing model using Streamlit.

## Dataset Details

Dataset: Bank Marketing Dataset

Number of Records: 45,211

Number of Features: 42 (after preprocessing)

Target Variable: y (Yes/No)

## Models Implemented

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

## Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------|---------|---------|---------|---------|---------|---------|
| Logistic Regression | 0.9012 | 0.9054 | 0.6440 | 0.3488 | 0.4525 | 0.4264 |
| Decision Tree | 0.8777 | 0.7135 | 0.4783 | 0.4991 | 0.4884 | 0.4191 |
| KNN | 0.8936 | 0.8084 | 0.5860 | 0.3091 | 0.4047 | 0.3742 |
| Naive Bayes | 0.8639 | 0.8088 | 0.4282 | 0.4877 | 0.4560 | 0.3797 |
| SVM | 0.9026 | 0.9053 | 0.6774 | 0.3195 | 0.4342 | 0.4213 |

## Observations

- SVM achieved the highest accuracy.
- Logistic Regression achieved the highest AUC score.
- Decision Tree achieved the highest recall.
- Naive Bayes provided competitive recall despite its simplicity.
- KNN achieved reasonable performance but lower recall.

## Best Model

Support Vector Machine (SVM)

Accuracy: 90.26%

## GitHub Repository

(Add GitHub Repository Link Here)

## Streamlit Application

(Add Streamlit Deployment Link Here)