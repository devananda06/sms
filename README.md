# SMS Spam Detector

A Streamlit app that classifies an SMS as spam or ham using TF-IDF + Linear SVM.

## Files
- `app.py`: Streamlit user interface
- `train_model.py`: downloads the UCI SMS Spam Collection dataset and trains the model
- `requirements.txt`: app dependencies
- `spam_model.joblib`: generated trained model required by the app

## Create the model
Run `python train_model.py` in an environment with internet access. Then upload `spam_model.joblib` to the same GitHub repository as `app.py`.

## Deploy
Deploy the repository on Streamlit Community Cloud and set the main file path to `app.py`.
