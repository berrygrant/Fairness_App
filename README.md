# Fairness in AI Toolkit

This repository provides a fully automated toolkit to help students evaluate and mitigate bias in machine learning datasets and models. The toolkit includes scripts and a Streamlit app that work with the COMPAS and StereoSet datasets.

## Contents

- `fairness_script_loader.py`: Loads and summarizes the dataset with fairness-related statistics.
- `baseline_model_trainer.py`: Trains a baseline logistic regression model and reports performance/fairness metrics.
- `fairness_intervention.py`: Applies fairness interventions (upsampling, downsampling, or reweighting) and retrains the model.
- `reflection_and_hypotheses.py`: Generates a structured reflection template based on student configuration.
- `fairness_app.py`: An interactive Streamlit web app to explore, model, and evaluate fairness metrics.

## How to Run

1. Install requirements:  
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the Streamlit app:  
   ```bash
   streamlit run fairness_app.py
   ```

## Dataset Requirements

Place the datasets in the appropriate paths before running:
- COMPAS: `compas-scores-two-years.csv`
- StereoSet: `bias-bench/stereoset/stereoset_dev.jsonl`
