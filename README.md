# Fairness in AI Toolkit

This repository provides a fully automated toolkit to help students evaluate and mitigate bias in machine learning datasets and models. The toolkit includes scripts and a Streamlit app that work with the COMPAS and StereoSet datasets.

## Contents

- `fairness_script_loader.py`: Loads and summarizes the dataset with fairness-related statistics.
- `baseline_model_trainer.py`: Trains a baseline logistic regression model and reports performance/fairness metrics.
- `fairness_intervention.py`: Applies fairness interventions (upsampling, downsampling, or reweighting) and retrains the model.
- `reflection_and_hypotheses.py`: Generates a structured reflection template based on student configuration.
- `fairness_app.py`: An interactive Streamlit web app to explore, model, and evaluate fairness metrics.

## How to Run

1. Clone or fork the repository  
   ```git 
   git clone https://github.com/berrygrant/Fairness_App.git     
   ``` 

2. Install requirements (preferably in a [virtual environment](https://docs.python.org/3/library/venv.html)):  
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit app on your local machine:  
   ```bash
   streamlit run fairness_app.py
   ```

Alternatively, you can access the deployed app: [https://fairness-in-ai-activity.streamlit.app/](https://fairness-in-ai-activity.streamlit.app/)  
**Note: Streamlit's free tier has limited memory resources, so the app may crash. I've set a cron job that will reboot the app every morning, but I will not reboot the app on an ad-hoc basis.**
