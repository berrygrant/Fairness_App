# fairness_script_loader.py (COMPAS and StereoSet)

import pandas as pd
import numpy as np
import os
import json

from sklearn.model_selection import train_test_split

# ==== CONFIGURATION SECTION ====
# Choose from: 'compas', 'stereoset'
dataset_name = 'stereoset'

if dataset_name == 'compas':
    data_path = 'compas-scores-two-years.csv'
    outcome_var = 'two_year_recid'
    sensitive_vars = ['age_cat', 'sex', 'race']
elif dataset_name == 'stereoset':
    data_path = 'bias-bench/stereoset/stereoset_dev.jsonl'
    outcome_var = 'ss'  # stereotypical score, can switch to 'lms' or 'icat' as needed
    sensitive_vars = ['bias_type']  # this will be 'gender', 'race', etc.
else:
    raise ValueError("Unknown dataset")

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
                record = {
                    'bias_type': item['bias_type'],
                    'ss': item.get('ss', np.nan),
                    'lms': item.get('lms', np.nan),
                    'icat': item.get('icat', np.nan)
                }
                data.append(record)
    return pd.DataFrame(data)

if dataset_name == 'compas':
    df = load_compas(data_path)
elif dataset_name == 'stereoset':
    df = load_stereoset(data_path)

# ==== BASIC STATS ====
print("\n===== DATA SUMMARY =====")
print(df[sensitive_vars + [outcome_var]].describe(include='all'))

# ==== CONTINGENCY-LIKE SUMMARY (for continuous outcomes) ====
def summarize_continuous_by_group(df, outcome_var, sensitive_vars):
    for var in sensitive_vars:
        print(f"\nGroup summaries for {var} vs {outcome_var}:")
        print(df.groupby(var)[outcome_var].describe())

# ==== CONTINGENCY TABLE (for COMPAS only) ====
def generate_contingency_tables(df, outcome_var, sensitive_vars):
    tables = {}
    for var in sensitive_vars:
        table = pd.crosstab(df[var], df[outcome_var], margins=True)
        tables[var] = table
    return tables

# ==== BALANCED ACCURACY ====
def balanced_accuracy_from_contingency(table):
    if 'All' in table.columns:
        table = table.drop(columns='All')
    if 'All' in table.index:
        table = table.drop(index='All')
    ba_scores = []
    for group in table.index:
        values = table.loc[group].values
        if len(values) == 2:
            total = values.sum()
            tpr = values[1] / total if total > 0 else 0
            tnr = values[0] / total if total > 0 else 0
            ba = 0.5 * (tpr + tnr)
            ba_scores.append((group, round(ba, 3)))
    return ba_scores

# ==== DISPLAY RESULTS ====
if dataset_name == 'compas':
    contingency_tables = generate_contingency_tables(df, outcome_var, sensitive_vars)
    for var, table in contingency_tables.items():
        print(f"\nContingency Table for {var} vs {outcome_var}:")
        print(table)
        ba_scores = balanced_accuracy_from_contingency(table)
        print(f"Balanced accuracy for {var}:")
        for group, score in ba_scores:
            print(f"  {group}: {score}")
elif dataset_name == 'stereoset':
    summarize_continuous_by_group(df, outcome_var, sensitive_vars)

# ==== MISSING DATA ====
print("\n===== MISSING DATA REPORT =====")
print(df[sensitive_vars + [outcome_var]].isnull().sum())

# ==== STRATIFIED SPLIT PREVIEW (Optional for modeling phase) ====
print("\n===== STRATIFIED SPLIT PREVIEW =====")
if dataset_name == 'compas':
    x_cols = [col for col in df.columns if col not in sensitive_vars + [outcome_var]]
    print(f"Available features for modeling (not including sensitive or outcome vars):\n{x_cols}")
elif dataset_name == 'stereoset':
    print("Sensitive groups present:", df['bias_type'].value_counts())
