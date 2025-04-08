import streamlit as st
import numpy as np
import joblib
import os
import gdown

# Google Drive model file ID (update this!)
MODEL_ID = 'YOUR_MODEL_FILE_ID'

# Download churn_model.pkl from Google Drive if not already present
model_path = 'churn_model.pkl'
if not os.path.exists(model_path):
    gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', model_path, quiet=False)

# Load model from downloaded file
model = joblib.load(model_path)

# Load scaler from local file (from GitHub)
scaler = joblib.load('scaler.pkl')

# Streamlit UI
st.set_page_config(page_title="Student Churn Predictor", layout="centered")
st.title("🎓 Student Churn Predictor")
st.markdown("Predict whether a student is likely to churn based on academic and behavioral data.")

# Input form
with st.form("churn_form"):
    attendance = st.slider("Class Attendance (%)", 0, 100, 75)
    percentage = st.slider("Test Percentage (%)", 0, 100, 65)
    homework_score = st.slider("Homework Score (1–5)", 1, 5, 3)
    tenth_percentage = st.slider("10th Board Percentage (%)", 0, 100, 70)

    income = st.selectbox("Family Income", ['Low', 'Mid', 'High'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    rating = st.selectbox("Student Rating", ['Poor', 'Average', 'Good', 'Very Good', 'Excellent'])

    submitted = st.form_submit_button("Predict Churn")

# Mappings
income_map = {'Low': 0, 'Mid': 1, 'High': 2}
gender_map = {'Male': 0, 'Female': 1}
rating_map = {'Poor': 0, 'Average': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}

if submitted:
    # Prepare input
    numerical = np.array([[attendance, percentage, homework_score, tenth_percentage]])
    categorical = np.array([[income_map[income], gender_map[gender], rating_map[rating]]])

    # Scale numerical part
    numerical_scaled = scaler.transform(numerical)

    # Combine both
    input_scaled = np.hstack((numerical_scaled, categorical))

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

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
