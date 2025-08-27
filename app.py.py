# app.py - Heart Disease Prediction App with Streamlit

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(
    page_title="❤️ CardioScan AI",
    page_icon="❤️",
    layout="centered"
)

# Title and description
st.title("❤️ CardioScan AI")
st.subheader("Heart Disease Risk Prediction")
st.write("""
This app predicts the likelihood of heart disease using machine learning.
Fill in the patient details below to get an instant risk assessment.
""")

# Load model, scaler, and feature names
@st.cache_resource
def load_model():
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_columns.pkl')
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Input Section
st.sidebar.header("📊 Patient Information")

age = st.sidebar.slider("Age", 20, 100, 50)

sex = st.sidebar.radio("Sex", ["Male", "Female"])

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
)

trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)

chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)

fbs = st.sidebar.checkbox("Fasting Blood Sugar > 120 mg/dl")

restecg = st.sidebar.selectbox(
    "Resting ECG",
    ["normal", "lv hypertrophy", "st-t abnormality"]
)

thalch = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)

exang = st.sidebar.checkbox("Exercise Induced Angina")

oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)

slope = st.sidebar.selectbox(
    "Slope of ST Segment",
    ["upsloping", "flat", "downsloping"]
)

ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)

thal = st.sidebar.selectbox(
    "Thalassemia",
    ["normal", "fixed defect", "reversable defect"]
)

# Preprocess input
input_data = pd.DataFrame({
    'age': [age],
    'sex': [1 if sex == 'Male' else 0],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [1 if fbs else 0],
    'thalch': [thalch],
    'exang': [1 if exang else 0],
    'oldpeak': [oldpeak],
    'ca': [ca]
})

# One-hot encode categorical features
cp_dummies = pd.get_dummies([cp], prefix='cp')
restecg_dummies = pd.get_dummies([restecg], prefix='restecg')
slope_dummies = pd.get_dummies([slope], prefix='slope')
thal_dummies = pd.get_dummies([thal], prefix='thal')

# Concatenate all
input_data = pd.concat([input_data, cp_dummies, restecg_dummies, slope_dummies, thal_dummies], axis=1)

# Reindex to match training data
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# Scale the data
try:
    input_scaled = scaler.transform(input_data)
except ValueError as e:
    st.error(f"Scaling error: {e}")
    st.stop()

# Predict
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# Display Result
st.markdown("---")
st.header("📋 Prediction Result")

if prediction == 1:
    st.markdown(f"<h3 style='color:red;'>⚠️ High Risk of Heart Disease</h3>", unsafe_allow_html=True)
    st.write(f"**Confidence:** {probability:.2%} chance of heart disease.")
else:
    st.markdown(f"<h3 style='color:green;'>✅ Low Risk of Heart Disease</h3>", unsafe_allow_html=True)
    st.write(f"**Confidence:** {1 - probability:.2%} chance of no heart disease.")

# Add disclaimer
st.markdown("---")
st.caption("""
*Disclaimer: This tool is for educational and demonstration purposes only. 
It is not a substitute for professional medical advice, diagnosis, or treatment.*
""")

# Optional: Show feature importance (if Random Forest)
if st.checkbox("Show Feature Importance"):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        st.bar_chart(feature_imp.head(10))
    else:
        st.write("Feature importance not available for this model.")