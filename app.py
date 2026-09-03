import streamlit as st
import pandas as pd
import pickle

# Set Page Config
st.set_page_config(page_title="Heart Disease Risk Assessment", layout="wide")

@st.cache_resource
def load_pipeline():
    with open("Classifier.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()

st.title("Heart Disease Cardiac Risk Assessment System")
st.write("Clinical Decision Support Tool powered by Machine Learning")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Patient Age", min_value=18, max_value=100, value=50)
    bp = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=220, value=120)
    chol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=200)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=24.5)
    trig = st.number_input("Triglyceride Level (mg/dL)", min_value=50, max_value=500, value=150)
    fbs = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=70, max_value=250, value=100)
    crp = st.number_input("C-Reactive Protein Level (mg/L)", min_value=0.0, max_value=20.0, value=3.8)

with col2:
    homo = st.number_input("Homocysteine Level (µmol/L)", min_value=0.0, max_value=30.0, value=10.0)
    sleep = st.number_input("Average Sleep Hours", min_value=2.0, max_value=14.0, value=7.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    exercise = st.selectbox("Exercise Habits", ["Low", "Medium", "High"])
    smoking = st.selectbox("Smoking Status", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High", "Unknown"])
    stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])

with col3:
    sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])
    diabetes = st.selectbox("Diabetes Diagnosis", ["No", "Yes"])
    family_hd = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
    high_bp = st.selectbox("High Blood Pressure Indicator", ["No", "Yes"])
    low_hdl = st.selectbox("Low HDL Cholesterol Indicator", ["No", "Yes"])
    high_ldl = st.selectbox("High LDL Cholesterol Indicator", ["No", "Yes"])

if st.button("Evaluate Cardiac Risk"):
    # Create DataFrame with exact feature names as training data
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Blood Pressure': bp,
        'Cholesterol Level': chol,
        'Exercise Habits': exercise,
        'Smoking': smoking,
        'Family Heart Disease': family_hd,
        'Diabetes': diabetes,
        'BMI': bmi,
        'High Blood Pressure': high_bp,
        'Low HDL Cholesterol': low_hdl,
        'High LDL Cholesterol': high_ldl,
        'Alcohol Consumption': alcohol,
        'Stress Level': stress,
        'Sleep Hours': sleep,
        'Sugar Consumption': sugar,
        'Triglyceride Level': trig,
        'Fasting Blood Sugar': fbs,
        'CRP Level': crp,
        'Homocysteine Level': homo
    }])

    # Predict probability and target class
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"**Assessment Outcome:** High Risk of Heart Disease (Positive)\n\n**Calculated Probability:** {probability:.2%}")
    else:
        st.success(f"**Assessment Outcome:** Low Risk of Heart Disease (Negative)\n\n**Calculated Probability:** {probability:.2%}")
