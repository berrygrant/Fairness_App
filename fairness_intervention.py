# fairness_intervention.py

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict

# ==== CONFIGURATION ====
dataset_name = 'compas'  # 'compas' or 'stereoset'
data_path = 'compas-scores-two-years.csv'  # or stereoset path
outcome_var = 'two_year_recid' if dataset_name == 'compas' else 'ss'
sensitive_var = 'race' if dataset_name == 'compas' else 'bias_type'
intervention_type = 'upsample'  # Options: 'upsample', 'downsample', 'reweight'

# ==== LOAD DATA ====
def load_compas(path):
    df = pd.read_csv(path)
    df = df[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30)]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']
    return df

def load_stereoset(path):
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

if dataset_name == 'compas':
    df = load_compas(data_path)
elif dataset_name == 'stereoset':
    df = load_stereoset(data_path)

# ==== CLEAN AND PREP ====
sensitive_values = df[sensitive_var].unique()
outcome_col = df[outcome_var].astype(int)
x_cols = [col for col in df.columns if col not in [outcome_var, sensitive_var]]
df = df.dropna(subset=[outcome_var] + x_cols)

X = pd.get_dummies(df[x_cols], drop_first=True)
y = df[outcome_var].astype(int)

df[sensitive_var] = df[sensitive_var].astype(str)  # for grouping

# ==== APPLY INTERVENTION ====
if intervention_type == 'upsample':
    frames = []
    for val in sensitive_values:
        subset = df[df[sensitive_var] == val]
        upsampled = resample(subset, replace=True, n_samples=df[sensitive_var].value_counts().max(), random_state=42)
        frames.append(upsampled)
    df_balanced = pd.concat(frames)
elif intervention_type == 'downsample':
    frames = []
    min_size = df[sensitive_var].value_counts().min()
    for val in sensitive_values:
        subset = df[df[sensitive_var] == val]
        downsampled = resample(subset, replace=False, n_samples=min_size, random_state=42)
        frames.append(downsampled)
    df_balanced = pd.concat(frames)
elif intervention_type == 'reweight':
    # No sampling, but compute weights
    df_balanced = df.copy()
    group_counts = df[sensitive_var].value_counts()
    df_balanced['sample_weight'] = df[sensitive_var].apply(lambda x: 1 / group_counts[x])
else:
    raise ValueError("Unsupported intervention type")

# ==== SPLIT AND MODEL ====
X_bal = pd.get_dummies(df_balanced[x_cols], drop_first=True)
y_bal = df_balanced[outcome_var].astype(int)
weights = df_balanced['sample_weight'] if 'sample_weight' in df_balanced.columns else None

X_train, X_test, y_train, y_test, sv_test = train_test_split(
    X_bal, y_bal, df_balanced[sensitive_var], test_size=0.3, stratify=df_balanced[sensitive_var], random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train, sample_weight=weights.loc[y_train.index] if weights is not None else None)
y_pred = model.predict(X_test_scaled)

# ==== METRICS ====
print("===== INTERVENTION RESULTS =====")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")

# ==== FAIRNESS ====
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

print("\n===== GROUP-WISE FAIRNESS METRICS =====")
gm = group_metrics(y_test, y_pred, sv_test)
for group, vals in gm.items():
    print(f"{group:12s} | TPR: {vals['TPR']:.2f} | FPR: {vals['FPR']:.2f} | PosRate: {vals['Positive Rate']:.2f}")
