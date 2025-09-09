import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import calibration_curve
from PIL import Image
# --- Load model & scaler ---
model = joblib.load("stacking_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Failure Risk Dashboard")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # --- Preprocessing ---
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["PATIENT", "time"])

    # Feature engineering
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    df["Pulse_Pressure"] = df["Systolic Blood Pressure"] - df["Diastolic Blood Pressure"]
    df["MAP"] = df["Diastolic Blood Pressure"] + (df["Pulse_Pressure"] / 3)

    expected_features = [
        "Body Weight",
        "Diastolic Blood Pressure",
        "Systolic Blood Pressure",
        "year",
        "month",
        "day",
        "hour",
        "Pulse_Pressure",
        "MAP"
    ]
    X = df[expected_features].fillna(df[expected_features].median())
    X_scaled = scaler.transform(X)

    # --- Predictions ---
    preds_proba = model.predict_proba(X_scaled)[:, 1]
    df["Risk Probability"] = preds_proba

    # --- Cohort View ---
    st.subheader("Cohort Risk Scores")
    st.dataframe(df[["PATIENT", "Risk Probability"]].groupby("PATIENT").mean().sort_values("Risk Probability", ascending=False))

    st.subheader("Risk Probability Distribution")
    st.bar_chart(df["Risk Probability"])

    # --- Evaluation Metrics (if true labels are available) ---
    if "TARGET" in df.columns:
        y_true = df["TARGET"]
        auroc = roc_auc_score(y_true, preds_proba)
        auprc = average_precision_score(y_true, preds_proba)
        st.write(f"**AUROC:** {auroc:.3f}, **AUPRC:** {auprc:.3f}")

        # Confusion matrix at 0.5 threshold
        y_pred = (preds_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        st.write("**Confusion Matrix:**")
        st.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

        # Calibration plot
        prob_true, prob_pred = calibration_curve(y_true, preds_proba, n_bins=10)
        fig, ax = plt.subplots()
        ax.plot(prob_pred, prob_true, marker='o')
        ax.plot([0,1],[0,1], linestyle='--', color='gray')
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Probability")
        ax.set_title("Calibration Curve")
        st.pyplot(fig)
    st.title("Explainability")

        # --- Description ---
    st.markdown("""
        This pipeline processes multiple heart failure time-series CSV files. It:

        - Combines the CSV files into a single dataset.
        - Cleans and sorts the records by patient and time.
        - Encodes categorical IDs.
        - Fills in missing values and scales numerical features.
        - Splits data into training and test sets.
        - Optionally balances class counts using SMOTE.
        - Trains several models:
        - Tree-based: Random Forest, XGBoost
        - Explainable AI using **ELI5** for model interpretability
        """)

        # --- Display image ---
    try:
        image = Image.open("image.png")
        st.image(image, caption="Pipeline Overview", use_container_width=True)
    except FileNotFoundError:
        st.error("image.png not found in the current folder!")