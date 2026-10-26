import streamlit as st
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "spam_model.joblib"

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

st.title("📩 SMS Spam Detector")
st.write("Enter an SMS message to check whether it looks like spam or a normal message.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.error("The trained model file (spam_model.joblib) is missing.")
    st.markdown(
        "Add `spam_model.joblib` to the same GitHub repository as `app.py`. "
        "You can generate it by running the training notebook/script first."
    )
else:
    message = st.text_area("SMS message", placeholder="Example: Congratulations! You have won a prize...")
    if st.button("Check message", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            prediction = int(model.predict([message])[0])
            if prediction == 1:
                st.error("🚨 This message looks like SPAM.")
            else:
                st.success("✅ This message looks like a normal message (HAM).")

st.caption("Demo project: TF-IDF + Linear SVM trained on the UCI SMS Spam Collection.")
