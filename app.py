import pandas as pd
import streamlit as st
import pickle
import numpy as np

# 页面基础配置
st.set_page_config(
    page_title="Heart Disease Risk Assessment System",
    page_icon="❤️",
    layout="wide",
)

# 加载模型
@st.cache_resource
def load_model():
    with open('Classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"模型加载失败，请确保 Classifier.pkl 已上传且格式正确。错误信息: {e}")

# 创建顶部的多标签页（和别人的一样！）
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Exploration", "Model Performance"])

with tab1:
    st.header("Cardiac Risk Prediction Panel")
    st.write("Please input the patient's clinical parameters below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input("Patient Age", min_value=18, max_value=100, value=50)
        Blood_Pressure = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=220, value=120)
        Cholesterol_Level = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=200)
        BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=24.5)
        Gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        Triglyceride_Level = st.number_input("Triglyceride Level (mg/dL)", min_value=50, max_value=500, value=150)
        Fasting_Blood_Sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=70, max_value=250, value=100)
        CRP_Level = st.number_input("C-Reactive Protein Level (mg/L)", min_value=0.0, max_value=20.0, value=3.0)
        Homocysteine_Level = st.number_input("Homocysteine Level (µmol/L)", min_value=0.0, max_value=30.0, value=10.0)
        Exercise_Habits = st.selectbox("Exercise Habits", ["Low", "Medium", "High"])

    with col3:
        Sleep_Hours = st.number_input("Average Sleep Hours", min_value=2, max_value=14, value=7)
        Smoking = st.selectbox("Smoking Status", ["No", "Yes"])
        Alcohol_Consumption = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High", "Unknown"])
        Stress_Level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
        Sugar_Consumption = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])

    # 附加风险项
    st.markdown("### Clinical History & Indicators")
    col4, col5 = st.columns(2)
    with col4:
        Diabetes = st.selectbox("Diabetes Diagnosis", ["No", "Yes"])
        Family_Heart_Disease = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
    with col5:
        High_Blood_Pressure = st.selectbox("High Blood Pressure Indicator", ["No", "Yes"])
        Low_HDL_Cholesterol = st.selectbox("Low HDL Cholesterol Indicator", ["No", "Yes"])
        High_LDL_Cholesterol = st.selectbox("High LDL Cholesterol Indicator", ["No", "Yes"])

    if st.button("Evaluate Cardiac Risk", type="primary"):
        # 构造 DataFrame 输入（必须与训练时的列名一致）
        input_data = pd.DataFrame({
            'Age': [Age],
            'Blood Pressure': [Blood_Pressure],
            'Cholesterol Level': [Cholesterol_Level],
            'BMI': [BMI],
            'Sleep Hours': [Sleep_Hours],
            'Triglyceride Level': [Triglyceride_Level],
            'Fasting Blood Sugar': [Fasting_Blood_Sugar],
            'CRP Level': [CRP_Level],
            'Homocysteine Level': [Homocysteine_Level],
            'Gender': [Gender],
            'Exercise Habits': [Exercise_Habits],
            'Smoking': [Smoking],
            'Family Heart Disease': [Family_Heart_Disease],
            'Diabetes': [Diabetes],
            'High Blood Pressure': [High_Blood_Pressure],
            'Low HDL Cholesterol': [Low_HDL_Cholesterol],
            'High LDL Cholesterol': [High_LDL_Cholesterol],
            'Alcohol Consumption': [Alcohol_Consumption],
            'Stress Level': [Stress_Level],
            'Sugar Consumption': [Sugar_Consumption]
        })

        try:
            prediction = model.predict(input_data)[0]
            if prediction == 1:
                st.error("Assessment Outcome: High Risk of Heart Disease (Positive)")
            else:
                st.success("Assessment Outcome: Low Risk of Heart Disease (Negative)")
        except Exception as e:
            st.error(f"Prediction failed due to feature mismatch. Details: {e}")

with tab2:
    st.header("Data Exploration")
    st.write("Exploratory Data Analysis and Dataset Overview")
    # 可以放一些概览指标（跟你的截图呼应）
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Features", "20")
    m2.metric("Target Variable", "Heart Disease Status")
    m3.metric("Model Used", "Machine Learning Pipeline")
    st.info("Upload your dataset preview or charts here if needed.")

with tab3:
    st.header("Model Performance")
    st.write("Algorithm evaluation metrics, confusion matrix, and ROC-AUC comparisons.")
    st.success("Trained using optimized Machine Learning algorithms with cross-validation.")
