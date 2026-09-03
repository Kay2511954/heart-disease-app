import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

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

# 创建多标签页
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Exploration", "Model Performance"])

# ==========================================
# TAB 1: 预测页面
# ==========================================
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

# ==========================================
# TAB 2: 数据探索图表展示 (Data Exploration)
# ==========================================
with tab2:
    st.header("Data Exploration & Visualizations")
    st.write("Exploratory Data Analysis of the Heart Disease Dataset")

    # 依次展示你第二章生成的图表（确保你的 GitHub 仓库里也上传了这些图片 png 文件）
    fig_files_ch2 = [
        ("Figure 2.1: Target Class Distribution", "figure_2_1_target_distribution.png"),
        ("Figure 2.2: Missing Value Counts per Feature", "figure_2_2_missing_values.png"),
        ("Figure 2.3: Histograms and KDE Plots for Continuous Biometric Features", "figure_2_3_numeric_distributions.png"),
        ("Figure 2.4: Correlation Matrix of Continuous Clinical Features", "figure_2_4_correlation_heatmap.png")
    ]

    for title, filename in fig_files_ch2:
        st.subheader(title)
        if os.path.exists(filename):
            st.image(filename, use_container_width=True)
        else:
            st.info(f"Image `{filename}` not uploaded yet. (You can generate and upload it from your Jupyter Notebook)")
        st.markdown("---")

# ==========================================
# TAB 3: 模型表现与评估图表展示 (Model Performance)
# ==========================================
with tab3:
    st.header("Model Performance & Evaluation")
    st.write("Confusion matrices, ROC curves, and algorithm comparison charts.")

    fig_files_ch4_5 = [
        ("Figure 4.1 & 4.2: Logistic Regression Evaluation", "fig_lr_eval.png"), # 如果有独立保存的图可以写在这里
        ("Figure 4.3 & 4.4: KNN Evaluation", "figure_4_4_knn_roc_curve.png"),
        ("Figure 4.5 & 4.6: Random Forest Evaluation", "fig_rf_roc_curve.png"),
        ("Figure 4.7 & 4.8: Gradient Boosting Evaluation", "fig_gb_roc_curve.png")
    ]

    for title, filename in fig_files_ch4_5:
        st.subheader(title)
        if os.path.exists(filename):
            st.image(filename, use_container_width=True)
        else:
            st.info(f"Image `{filename}` not uploaded yet.")
        st.markdown("---")
