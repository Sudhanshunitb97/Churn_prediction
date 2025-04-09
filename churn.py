import streamlit as st
import numpy as np
import joblib
import os
import gdown

# ✅ Google Drive File ID for your churn_model.pkl
MODEL_ID = '1EXqTXVLADlqyPbXkwrTix3yN3vm6_89X'  # ← REPLACE THIS
model_url = f'https://drive.google.com/uc?id={MODEL_ID}'
model_path = 'churn_model.pkl'

# ✅ Download the model if not already present
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# ✅ Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load('scaler.pkl')  # This should be in your GitHub repo

# ✅ Streamlit UI
st.set_page_config(page_title="Student Churn Predictor", layout="centered")
st.title("🎓 Student Churn Predictor")
st.markdown("Predict whether a student is likely to churn based on academic and behavioral data.")

# 🔘 Input form
with st.form("churn_form"):
    attendance = st.slider("Class Attendance (%)", 0, 100, 75)
    percentage = st.slider("Test Percentage (%)", 0, 100, 65)
    homework_score = st.slider("Homework Score (1–5)", 1, 5, 3)
    tenth_percentage = st.slider("10th Board Percentage (%)", 0, 100, 70)

    income = st.selectbox("Family Income", ['Low', 'Mid', 'High'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    rating = st.selectbox("Student Rating", ['Poor', 'Average', 'Good', 'Very Good', 'Excellent'])

    submitted = st.form_submit_button("Predict Churn")

# 🔠 Mappings
income_map = {'Low': 0, 'Mid': 1, 'High': 2}
gender_map = {'Male': 0, 'Female': 1}
rating_map = {'Poor': 0, 'Average': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}

# 🔮 Prediction
if submitted:
    numeric_features = np.array([[attendance, percentage, homework_score, tenth_percentage]])
    categorical_features = np.array([[income_map[income], gender_map[gender], rating_map[rating]]])

    numeric_scaled = scaler.transform(numeric_features)
    input_features = np.hstack((numeric_scaled, categorical_features))  # shape = (1, 7)

    prediction = model.predict(input_features)[0]
    prob = model.predict_proba(input_features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn — Probability: {prob:.2%}")
    else:
        st.success(f"✅ Likely to Stay — Churn Probability: {prob:.2%}")

    st.markdown("### 💡 Top 3 Features Driving Prediction")
    st.markdown("""
    1. **Attendance** – Lower class attendance often signals disengagement  
    2. **Homework Score** – Inconsistent homework effort is tied to higher churn risk  
    3. **Student Rating** – Self-assessed low satisfaction tends to align with churn
    """)
    st.caption("Based on feature importance from the Random Forest model.")
