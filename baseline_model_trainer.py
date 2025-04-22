# baseline_model_trainer.py

import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict

# ==== CONFIGURATION SECTION ====
dataset_name = 'compas'  # or 'stereoset'
data_path = 'compas-scores-two-years.csv'  # or stereoset path
outcome_var = 'two_year_recid' if dataset_name == 'compas' else 'ss'
sensitive_vars = ['age_cat', 'sex', 'race'] if dataset_name == 'compas' else ['bias_type']

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
                label = 1 if item['ss'] > 50 else 0  # threshold for stereotype bias
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

# ==== FEATURE SELECTION ====
x_cols = [col for col in df.columns if col not in sensitive_vars + [outcome_var]]
df = df.dropna(subset=[outcome_var] + x_cols)

X = pd.get_dummies(df[x_cols], drop_first=True)
y = df[outcome_var].astype(int)

# ==== STRATIFIED SPLIT ====
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df[sensitive_vars], test_size=0.3, random_state=42, stratify=df[sensitive_vars[0]])

# ==== TRAIN MODEL ====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# ==== CLASSIFICATION METRICS ====
print("===== OVERALL PERFORMANCE =====")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")

# ==== FAIRNESS METRICS ====
def group_metrics(y_true, y_pred, group_labels):
    results = defaultdict(dict)
    for var in group_labels.columns:
        for group in group_labels[var].unique():
            idx = group_labels[var] == group
            if sum(idx) == 0:
                continue
            yt, yp = y_true[idx], y_pred[idx]
            results[var][group] = {
                'TPR': np.sum((yt == 1) & (yp == 1)) / max(np.sum(yt == 1), 1),
                'FPR': np.sum((yt == 0) & (yp == 1)) / max(np.sum(yt == 0), 1),
                'Positive Rate': np.mean(yp)
            }
    return results

print("\n===== FAIRNESS METRICS BY GROUP =====")
fairness = group_metrics(y_test.values, y_pred, df_test)
for attr, scores in fairness.items():
    print(f"\n{attr}:")
    for group, vals in scores.items():
        tpr = vals['TPR']
        fpr = vals['FPR']
        pr  = vals['Positive Rate']
        print(f"  {group:12s} | TPR: {tpr:.2f} | FPR: {fpr:.2f} | Positive Rate: {pr:.2f}")
