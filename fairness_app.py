import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Fairness in AI Toolkit", layout="wide")
st.title("🔍 Fairness in AI: Automated Assignment Toolkit")

# ==== Dataset Selection ====
st.sidebar.header("Step 1: Dataset Setup")
dataset_name = st.sidebar.selectbox("Choose dataset", ["compas", "stereoset"])
intervention = st.sidebar.selectbox("Choose fairness intervention", ["none", "upsample", "downsample", "reweight"])

if dataset_name == "compas":
    path = './data/compas-scores-two-years.csv'
    sensitive_var = st.sidebar.selectbox("Sensitive attribute", ["age_cat", "sex", "race"])
    outcome_var = "two_year_recid"
elif dataset_name == "stereoset":
    path = ['./data/stereoset_dev.json', './data/stereoset_test.json']
    sensitive_var = "bias_type"
    outcome_var = "label"


def load_data():
    if dataset_name == "compas":
        df = pd.read_csv(path)
        df = df[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30)]
        df = df[df['is_recid'] != -1]
        df = df[df['c_charge_degree'] != 'O']
        df = df[df['score_text'] != 'N/A']
        return df
    elif dataset_name == "stereoset":
        combined = []
        for fname in ["./data/stereoset_dev.json", "./data/stereoset_test.json"]:
            with open(fname, "r") as f:
                data_json = json.load(f)
            for item in data_json["data"]["intersentence"]:
                bias_type = item["bias_type"]
                target = item["target"]
                for s in item["sentences"]:
                    gold_label = s.get("gold_label", "").strip().lower()
                    if gold_label in ["stereotype", "anti-stereotype"]:
                        label = 1 if gold_label == "stereotype" else 0
                        combined.append({
                            "target": target,
                            "bias_type": bias_type,
                            "sentence": s["sentence"],
                            "gold_label": gold_label,
                            "label": label
                        })
    return pd.DataFrame(combined)

# ==== Load and Show Data ====

# Add expandable explanation of interventions
# Show outcome variable
st.sidebar.markdown(f"**Outcome Variable:** `{outcome_var}`")

# Dynamic description of selected intervention
intervention_descriptions = {
    "none": (
        "**None**\n\n"
        "No resampling is performed. The model is trained on the dataset as-is, regardless of group imbalances."
    ),
    "upsample": (
        "**Upsample**\n\n"
        "Each group in the sensitive variable is increased in size to match the **largest group**. This is done by randomly duplicating existing rows (with replacement), ensuring all groups have equal sample size."
    ),
    "downsample": (
        "**Downsample**\n\n"
        "Each group is reduced in size to match the **smallest group**. This is done by randomly removing rows (without replacement) so all groups are equally represented — but with fewer total samples."
    ),
    "reweight": (
        "**Reweight**\n\n"
        "Each row is assigned a weight inversely proportional to the size of its group. Underrepresented groups are given more influence during model training without duplicating or removing data."
    ),
}

# Show description below the dropdown
st.sidebar.markdown("### 📖 Intervention Description")
st.sidebar.info(intervention_descriptions.get(intervention, "No description available."))

df = load_data()
available_features = [col for col in df.columns if col not in [outcome_var, sensitive_var]]

if st.checkbox("Preview raw data"):
    st.dataframe(df.head(n=20))

model_name = st.sidebar.selectbox(
    "Choose a classification model:",
    ["Logistic Regression", "Random Forest", "Support Vector Machine", "K-Nearest Neighbors"]
)

st.sidebar.markdown("### ⚙️ Hyperparameters")

if model_name == "Logistic Regression":
    C = st.sidebar.slider("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0)
    max_iter = st.sidebar.slider("Max Iterations", 100, 2000, 1000)

elif model_name == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 300, 100)
    max_depth = st.sidebar.slider("Max Depth (0 = None)", 0, 50, 0)

elif model_name == "Support Vector Machine":
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])

elif model_name == "K-Nearest Neighbors":
    n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, 5)

# ==== Fairness Intervention ====
st.sidebar.header("Step 2: Run Analysis")
selection_mode = st.sidebar.radio("Select predictors:", ["All variables", "Manual selection"])

if selection_mode == "Manual selection":
    selected_vars = st.sidebar.multiselect(
        "Select predictor variables to include in the model:",
        options=available_features,
        default=[]
    )
else:
    selected_vars = available_features

# Keep only predictors with sufficient non-null values
min_non_null = int(0.8 * len(df))  # keep columns with at least 80% non-null
selected_vars = [col for col in selected_vars if df[col].notnull().sum() >= min_non_null]

# Show summary of non-null values for selected predictors
if selected_vars:
    non_null_summary = pd.DataFrame({
        "Feature": selected_vars,
        "Non-null Count": [df[col].notnull().sum() for col in selected_vars],
        "Total Rows": len(df),
        "Percent Non-null": [f"{100 * df[col].notnull().mean():.1f}%" for col in selected_vars]
    })
    st.subheader("📊 Feature Availability Summary")
    st.dataframe(non_null_summary)

if not selected_vars:
    st.error("All selected predictors have too many missing values. Try switching to manual selection or remove highly sparse columns.")
    st.stop()

run_analysis = st.sidebar.button("Run Model and Evaluate")

if run_analysis:
    df[sensitive_var] = df[sensitive_var].astype(str)

    if not selected_vars:
        st.error("You must select at least one predictor variable to proceed.")
        st.stop()

    x_cols = selected_vars
    df = df.dropna(subset=[outcome_var] + selected_vars)

    frames = []
    group_sizes = df[sensitive_var].value_counts()

    for val in group_sizes.index:
        subset = df[df[sensitive_var] == val]
        if subset.empty:
            continue

        if intervention == 'upsample':
            target_n = group_sizes.max()
            resampled = resample(subset, replace=True, n_samples=target_n, random_state=42)
            frames.append(resampled)

        elif intervention == 'downsample':
            target_n = group_sizes.min()
            if len(subset) >= target_n:
                resampled = resample(subset, replace=False, n_samples=target_n, random_state=42)
                frames.append(resampled)

        else:
            frames.append(subset)

    if frames:
        df_bal = pd.concat(frames)
    elif intervention == 'reweight':
        df_bal = df.copy()
        counts = df[sensitive_var].value_counts()
        df_bal['sample_weight'] = df[sensitive_var].apply(lambda x: 1 / counts[x])
    else:
        df_bal = df.copy()
        st.warning("No groups were resampled, using original data.")

    if df_bal.empty:
        st.error("No data available after resampling. Try a different intervention or inspect your sensitive variable.")
        st.stop()

    X = pd.get_dummies(df_bal[x_cols], drop_first=True)
    y = df_bal[outcome_var].astype(int)
    weights = df_bal['sample_weight'] if 'sample_weight' in df_bal.columns else None

    group_counts = Counter(df_bal[sensitive_var])
    if len(group_counts) < 2 or min(group_counts.values()) < 2:
        st.warning("Not enough data to stratify by the selected sensitive variable. Proceeding without stratification.")
        stratify_arg = None
    else:
        stratify_arg = df_bal[sensitive_var]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=stratify_arg, random_state=42
    )
    sv_train, sv_test = train_test_split(
        df_bal[sensitive_var], test_size=0.3, stratify=stratify_arg, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model selection
    if model_name == "Logistic Regression":
        model = LogisticRegression(C=C, max_iter=max_iter)

    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None if max_depth == 0 else max_depth,
            random_state=42
        )

    elif model_name == "Support Vector Machine":
        model = SVC(C=C, kernel=kernel, probability=True)

    elif model_name == "K-Nearest Neighbors":
        model = KNeighborsClassifier(n_neighbors=n_neighbors)


# Fit model with or without sample weights depending on model type
    if model_name in ["Logistic Regression", "Random Forest"]:
        model.fit(X_train_scaled, y_train, sample_weight=weights.loc[y_train.index] if weights is not None else None)
    else:
        model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    st.markdown(f"---")
    st.markdown(f"### 🔧 Model Run: {model_name}")
    st.markdown("#### Hyperparameter Configuration")
    if model_name == "Logistic Regression":
        st.markdown(f"- **C:** {C}")
        st.markdown(f"- **Max Iterations:** {max_iter}")
    elif model_name == "Random Forest":
        st.markdown(f"- **n_estimators:** {n_estimators}")
        st.markdown(f"- **max_depth:** {max_depth if max_depth != 0 else 'None'}")
    elif model_name == "Support Vector Machine":
        st.markdown(f"- **C:** {C}")
        st.markdown(f"- **Kernel:** {kernel}")
    elif model_name == "K-Nearest Neighbors":
        st.markdown(f"- **n_neighbors:** {n_neighbors}")

    st.markdown(f"---")
    st.subheader("📈 Model Performance")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.3f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred):.3f}")
    st.write(f"**Recall:** {recall_score(y_test, y_pred):.3f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.3f}")

    def group_metrics(y_true, y_pred, group_labels):
        results = {}
        for group in group_labels.unique():
            idx = group_labels == group
            yt, yp = y_true[idx], y_pred[idx]
            results[group] = {
                'TPR (True Positive Rate)': np.sum((yt == 1) & (yp == 1)) / max(np.sum(yt == 1), 1),
                'FPR (False Positive Rate)': np.sum((yt == 0) & (yp == 1)) / max(np.sum(yt == 0), 1),
                'Positive Rate': np.mean(yp)
            }
        return results

    gm = group_metrics(y_test, y_pred, sv_test)
    st.subheader("⚖️ Fairness by Sensitive Group")
    fairness_df = pd.DataFrame(gm).T.reset_index().rename(columns={'index': sensitive_var})
    st.dataframe(fairness_df)

    st.download_button("Download Fairness Summary", data=fairness_df.to_csv(index=False), file_name="fairness_metrics.csv")
