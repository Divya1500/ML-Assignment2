import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="Term Deposit Predictor",
    layout="wide"
)

MODEL_FILES = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "SVM": "model/svm.pkl",
}

TARGET = "y"
N_FEATURES = 42


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def prepare_target(series):
    """Convert target column to 0/1"""

    if series.dtype == bool:
        return series.astype(int)

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "yes": 1,
                "no": 0,
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
            }
        )
    )

    if mapped.isna().any():
        raise ValueError(
            "Target column y contains unsupported values."
        )

    return mapped.astype(int)


st.title(
    "Bank Marketing: Will the Client Subscribe to a Term Deposit?"
)

st.caption(
    "Compare five trained classification models using the uploaded test CSV."
)

with st.sidebar:

    st.header("Controls")

    model_choice = st.selectbox(
        "Choose a Model",
        list(MODEL_FILES.keys()),
    )

    uploaded_file = st.file_uploader(
        "Upload Test CSV",
        type=["csv"],
    )

    st.info(
        "Use test_data.csv provided in the repository."
    )

if uploaded_file is None:
    st.info("Please upload a CSV file.")
    st.stop()

try:

    df = pd.read_csv(uploaded_file)

    if TARGET not in df.columns:
        raise ValueError(
            "CSV must contain a target column named 'y'."
        )

    X = df.drop(columns=[TARGET])

    if X.shape[1] != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} features, found {X.shape[1]}."
        )

    y = prepare_target(df[TARGET])

    model = load_model(
        MODEL_FILES[model_choice]
    )

    pred = model.predict(X.values)

    proba = model.predict_proba(X.values)[:, 1]

except Exception as e:

    st.error(f"Error: {e}")
    st.stop()

st.subheader(
    f"Results for {model_choice}"
)

metrics = {
    "Accuracy": accuracy_score(y, pred),
    "AUC": roc_auc_score(y, proba),
    "Precision": precision_score(
        y,
        pred,
        zero_division=0,
    ),
    "Recall": recall_score(
        y,
        pred,
        zero_division=0,
    ),
    "F1": f1_score(
        y,
        pred,
        zero_division=0,
    ),
    "MCC": matthews_corrcoef(y, pred),
}

cols = st.columns(len(metrics))

for col, (metric_name, value) in zip(
    cols,
    metrics.items(),
):
    col.metric(
        metric_name,
        f"{value:.4f}",
    )

left, right = st.columns(2)

with left:

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(
        figsize=(4, 4)
    )

    ConfusionMatrixDisplay(
        confusion_matrix(y, pred),
        display_labels=["No", "Yes"],
    ).plot(
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )

    st.pyplot(fig)

with right:

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y,
        pred,
        target_names=["No", "Yes"],
        output_dict=True,
        zero_division=0,
    )

    st.dataframe(
        pd.DataFrame(report)
        .T
        .round(3)
    )

st.subheader(
    "Comparison of All Models"
)

rows = []

for name, path in MODEL_FILES.items():

    current_model = load_model(path)

    p = current_model.predict(X.values)

    pr = current_model.predict_proba(X.values)[:, 1]

    rows.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y, p),
            "AUC": roc_auc_score(y, pr),
            "Precision": precision_score(
                y,
                p,
                zero_division=0,
            ),
            "Recall": recall_score(
                y,
                p,
                zero_division=0,
            ),
            "F1": f1_score(
                y,
                p,
                zero_division=0,
            ),
            "MCC": matthews_corrcoef(y, p),
        }
    )

st.dataframe(
    pd.DataFrame(rows)
    .set_index("Model")
    .round(4)
)
