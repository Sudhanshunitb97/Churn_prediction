import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('churn_model_compressed_2.pkl')        # RandomForestClassifier
scaler = joblib.load('scaler.pkl')            # StandardScaler fitted on 4 numeric columns

# Set page config
st.set_page_config(page_title="Student Churn Predictor", layout="centered")
st.title("🎓 Student Churn Predictor")
st.markdown("Predict whether a student is likely to churn based on performance and behavioral inputs.")

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

# Category mappings
income_map = {'Low': 0, 'Mid': 1, 'High': 2}
gender_map = {'Male': 0, 'Female': 1}
rating_map = {'Poor': 0, 'Average': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}

# Prepare input
numerical = np.array([[attendance, percentage, homework_score, tenth_percentage]])
categorical = np.array([[income_map[income], gender_map[gender], rating_map[rating]]])

# Scale only numerical part
numerical_scaled = scaler.transform(numerical)

# Combine scaled numerical + raw categorical
input_scaled = np.hstack((numerical_scaled, categorical))

# Predict
if submitted:
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn — Probability: {prob:.2%}")
    else:
        st.success(f"✅ Likely to Stay — Churn Probability: {prob:.2%}")

    # Explain top 3 features
    st.markdown("### 💡 Top 3 Features Driving Churn Prediction")
    st.markdown("""
    1. **Attendance** – Lower class attendance often signals disengagement  
    2. **Homework Score** – Inconsistent homework effort is tied to higher churn risk  
    3. **Student Rating** – Self-assessed low satisfaction tends to align with churn
    """)
    st.caption("Based on feature importance from the Random Forest model.")