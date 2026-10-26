"""Train the SMS spam model and save spam_model.joblib.

Run locally or in Google Colab:
    python train_model.py

The script downloads the public UCI SMS Spam Collection dataset.
"""
import io
import urllib.request
import zipfile

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def load_sms():
    with urllib.request.urlopen(URL, timeout=30) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        with archive.open("SMSSpamCollection") as f:
            return pd.read_csv(f, sep="\\t", names=["label", "text"])

def main():
    df = load_sms().dropna().drop_duplicates().reset_index(drop=True)
    X = df["text"].astype(str)
    y = (df["label"].str.lower() == "spam").astype(int)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("clf", LinearSVC(random_state=42)),
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, "spam_model.joblib")
    print(f"Saved spam_model.joblib using {len(X_train)} training messages.")

if __name__ == "__main__":
    main()
