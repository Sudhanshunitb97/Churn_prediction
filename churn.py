import streamlit as st
import numpy as np
import joblib
import os
import gdown

# 🔹 Google Drive Model ID
MODEL_ID = '1cQxDwCBZqp_4E1dtjdYZ3Vo8LUUohHkD'
model_url = f'https://drive.google.com/uc?id={MODEL_ID}'
model_path = 'churn_model.pkl'

# 🔽 Download model if not already downloaded
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# 🔹 Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load("scaler.pkl")  # scaler.pkl should be present in the GitHub repo

# 🖼️ UI setup
st.set_page_config(page_title="Student Churn Predictor", layout="centered")
st.title("🎓 Student Churn Predictor")
st.markdown("Use academic and behavioral features to predict if a student is likely to churn.")

# 🧾 Input form
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    income = st.selectbox("Family Income", ["Low", "Mid", "High", "Unknown"])
    discipline = st.slider("Discipline Score (1–5)", 1, 5, 3)
    hw_score = st.slider("Homework Score (1–5)", 1, 5, 3)

    attendance = st.slider("Class Attendance (%)", 0, 100, 75)
    test_avg = st.slider("Test Average Score (%)", 0, 100, 60)
    rating = st.slider("Student Rating (1–10)", 1, 10, 7)
    tenth = st.slider("10th Percentage", 0, 100, 70)

    submitted = st.form_submit_button("Predict Churn")

# 🔠 Encoders
gender_map = {"Female": 0, "Male": 1, "Other": 2}
income_map = {"Low": 0, "Mid": 1, "High": 2, "Unknown": -1}

# 🧮 Prediction
if submitted:
    cat_feats = np.array([[gender_map[gender], income_map[income], discipline, hw_score]])
    num_feats = np.array([[attendance, test_avg, rating, tenth]])
    num_scaled = scaler.transform(num_feats)

    input_features = np.hstack((cat_feats, num_scaled))  # final shape (1, 8)

    prediction = model.predict(input_features)[0]
    prob = model.predict_proba(input_features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn — Probability: {prob:.2%}")
    else:
        st.success(f"✅ Likely to Stay — Churn Probability: {prob:.2%}")

    st.markdown("### 💡 Top Influential Factors")
    st.markdown("""
    1. **Attendance** — Poor attendance strongly linked with churn  
    2. **Homework & Discipline** — Less effort = higher risk  
    3. **Student Rating** — Lower satisfaction = more likely to churn
    """)
