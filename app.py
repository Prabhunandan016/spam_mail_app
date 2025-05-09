import streamlit as st
import joblib
import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set Streamlit page config
st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("Spam Message Classifier")

# Load model
model_path = "spam_model.pkl"
if not os.path.exists(model_path):
    st.error("Model file not found. Please run spam_classifier.py to train and save the model.")
    st.stop()

model = joblib.load(model_path)

# Load test data for metrics (from mail_data.csv)
@st.cache_data
def load_data():
    df = pd.read_csv("mail_data.csv", encoding="latin-1")
    for col in df.columns:
        if "label" in col.lower() or "category" in col.lower() or "v1" in col.lower():
            label_col = col
        if "message" in col.lower() or "text" in col.lower() or "v2" in col.lower():
            message_col = col
    df = df[[label_col, message_col]]
    df.columns = ["Label", "Message"]
    df["Label"] = df["Label"].map({"ham": 0, "spam": 1})
    return df

df = load_data()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Label"], test_size=0.2, random_state=42, stratify=df["Label"])
y_pred = model.predict(X_test)

# Input message
st.write("Enter a message below to check if it's **Spam** or **Ham**.")
message = st.text_area("Message", height=150)

if st.button("Classify"):
    if message.strip() == "":
        st.warning("Please enter a message before classifying.")
    else:
        prob = model.predict_proba([message])[0][1]
        prediction = "SPAM" if prob >= 0.5 else "HAM"

        st.subheader("Prediction Result")
        st.success(f"Message is classified as: **{prediction}**")
        st.info(f"Spam Probability: {prob:.2f}")

# Display evaluation metrics
st.subheader("Model Evaluation on Test Data")
col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    st.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")

with col2:
    st.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    st.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

# Confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)
