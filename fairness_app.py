# fairness_app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict

st.set_page_config(page_title="Fairness in AI Toolkit", layout="wide")
st.title("🔍 Fairness in AI: Automated Assignment Toolkit")

# ==== Dataset Selection ====
st.sidebar.header("Step 1: Dataset Setup")
dataset_name = st.sidebar.selectbox("Choose dataset", ["compas", "stereoset"])
intervention = st.sidebar.selectbox("Choose fairness intervention", ["none", "upsample", "downsample", "reweight"])

if dataset_name == "compas":
    path = "compas-scores-two-years.csv"
    sensitive_var = st.sidebar.selectbox("Sensitive attribute", ["age_cat", "sex", "race"])
    outcome_var = "two_year_recid"
    def load_data():
        df = pd.read_csv(path)
        df = df[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30)]
        df = df[df['is_recid'] != -1]
        df = df[df['c_charge_degree'] != 'O']
        df = df[df['score_text'] != 'N/A']
        return df
else:
    path = "bias-bench/stereoset/stereoset_dev.jsonl"
    sensitive_var = "bias_type"
    outcome_var = "ss"
    def load_data():
        data = []
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'bias_type' in item and 'ss' in item:
                    label = 1 if item['ss'] > 50 else 0
                    record = {
                        'bias_type': item['bias_type'],
                        'ss': label,
                        'lms': item.get('lms', np.nan),
                        'icat': item.get('icat', np.nan)
                    }
                    data.append(record)
        return pd.DataFrame(data)

# ==== Load and Show Data ====
df = load_data()
if st.checkbox("Preview raw data"):
    st.dataframe(df.head())

# ==== Fairness Intervention ====
st.sidebar.header("Step 2: Run Analysis")
run_analysis = st.sidebar.button("Run Model and Evaluate")

if run_analysis:
    df[sensitive_var] = df[sensitive_var].astype(str)
    x_cols = [col for col in df.columns if col not in [outcome_var, sensitive_var]]
    df = df.dropna(subset=[outcome_var] + x_cols)

    if intervention == 'upsample':
        frames = []
        for val in df[sensitive_var].unique():
            subset = df[df[sensitive_var] == val]
            upsampled = resample(subset, replace=True, n_samples=df[sensitive_var].value_counts().max(), random_state=42)
            frames.append(upsampled)
        df_bal = pd.concat(frames)
    elif intervention == 'downsample':
        frames = []
        for val in df[sensitive_var].unique():
            subset = df[df[sensitive_var] == val]
            downsampled = resample(subset, replace=False, n_samples=df[sensitive_var].value_counts().min(), random_state=42)
            frames.append(downsampled)
        df_bal = pd.concat(frames)
    elif intervention == 'reweight':
        df_bal = df.copy()
        counts = df[sensitive_var].value_counts()
        df_bal['sample_weight'] = df[sensitive_var].apply(lambda x: 1 / counts[x])
    else:
        df_bal = df.copy()

    X = pd.get_dummies(df_bal[x_cols], drop_first=True)
    y = df_bal[outcome_var].astype(int)
    weights = df_bal['sample_weight'] if 'sample_weight' in df_bal.columns else None

    X_train, X_test, y_train, y_test, sv_test = train_test_split(
        X, y, df_bal[sensitive_var], test_size=0.3, stratify=df_bal[sensitive_var], random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train, sample_weight=weights.loc[y_train.index] if weights is not None else None)
    y_pred = model.predict(X_test_scaled)

    # Metrics
    st.subheader("📈 Model Performance")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.3f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred):.3f}")
    st.write(f"**Recall:** {recall_score(y_test, y_pred):.3f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.3f}")

    def group_metrics(y_true, y_pred, group_labels):
        results = defaultdict(dict)
        for group in group_labels.unique():
            idx = group_labels == group
            yt, yp = y_true[idx], y_pred[idx]
            results[group] = {
                'TPR': np.sum((yt == 1) & (yp == 1)) / max(np.sum(yt == 1), 1),
                'FPR': np.sum((yt == 0) & (yp == 1)) / max(np.sum(yt == 0), 1),
                'Positive Rate': np.mean(yp)
            }
        return results

    gm = group_metrics(y_test, y_pred, sv_test)
    st.subheader("⚖️ Fairness by Sensitive Group")
    fairness_df = pd.DataFrame(gm).T.reset_index().rename(columns={'index': sensitive_var})
    st.dataframe(fairness_df)

    st.download_button("Download Fairness Summary", data=fairness_df.to_csv(index=False), file_name="fairness_metrics.csv")
