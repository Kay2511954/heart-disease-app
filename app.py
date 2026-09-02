"""
Heart Disease Cardiac Risk Assessment System
Streamlit deployment for the BMDS2003 Data Science group project.

Loads the trained Logistic Regression pipeline (Classifier.pkl) and provides
the same 21-field clinical input form described in the report's Figure 6.3,
using the full sklearn pipeline (preprocessing + model) for prediction so the
one-hot encoding always matches what the model was trained on.
"""

import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Cardiac Risk Assessment", layout="centered")


@st.cache_resource
def load_model():
    with open("Classifier.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["num_cols"], bundle["cat_cols"], bundle["train_means"]


model, num_cols, cat_cols, train_means = load_model()


def predict(input_dict):
    """Build a one-row DataFrame in the same column order the pipeline expects
    and run it through the full trained pipeline (preprocessing + model)."""
    row = pd.DataFrame([input_dict])[num_cols + cat_cols]
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]
    label = "High Risk of Heart Disease (Positive)" if pred == 1 else "Low Risk of Heart Disease (Negative)"
    return label, proba


def main():
    st.title("Heart Disease Cardiac Risk Assessment System")
    st.write("Clinical Decision Support Tool powered by Machine Learning")

    st.info(
        "This tool uses a Logistic Regression model trained on the group's "
        "10,000-record clinical dataset. Model performance is documented in "
        "Section 4.1 of the project report."
    )

    st.subheader("Clinical & Biometric Inputs")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Patient Age", min_value=18, max_value=100, value=int(round(train_means["Age"])))
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=220, value=int(round(train_means["Blood Pressure"])))
        cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=int(round(train_means["Cholesterol Level"])))
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=round(float(train_means["BMI"]), 1))
        triglyceride = st.number_input("Triglyceride Level (mg/dL)", min_value=50, max_value=500, value=int(round(train_means["Triglyceride Level"])))

    with col2:
        fasting_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=70, max_value=250, value=int(round(train_means["Fasting Blood Sugar"])))
        crp = st.number_input("C-Reactive Protein Level (mg/L)", min_value=0.0, max_value=20.0, value=round(float(train_means["CRP Level"]), 1))
        homocysteine = st.number_input("Homocysteine Level (\u00b5mol/L)", min_value=0.0, max_value=30.0, value=round(float(train_means["Homocysteine Level"]), 1))
        sleep_hours = st.number_input("Average Sleep Hours", min_value=2, max_value=14, value=int(round(train_means["Sleep Hours"])))

    st.subheader("Lifestyle & Medical History")
    col3, col4 = st.columns(2)

    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])
        exercise = st.selectbox("Exercise Habits", ["Low", "Medium", "High"])
        smoking = st.selectbox("Smoking Status", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High", "Unknown"])
        stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
        sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])

    with col4:
        diabetes = st.selectbox("Diabetes Diagnosis", ["No", "Yes"])
        family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
        high_bp = st.selectbox("High Blood Pressure Indicator", ["No", "Yes"])
        low_hdl = st.selectbox("Low HDL Cholesterol Indicator", ["No", "Yes"])
        high_ldl = st.selectbox("High LDL Cholesterol Indicator", ["No", "Yes"])

    if st.button("Evaluate Cardiac Risk"):
        input_dict = {
            "Age": age,
            "Blood Pressure": blood_pressure,
            "Cholesterol Level": cholesterol,
            "BMI": bmi,
            "Sleep Hours": sleep_hours,
            "Triglyceride Level": triglyceride,
            "Fasting Blood Sugar": fasting_sugar,
            "CRP Level": crp,
            "Homocysteine Level": homocysteine,
            "Gender": gender,
            "Exercise Habits": exercise,
            "Smoking": smoking,
            "Family Heart Disease": family_history,
            "Diabetes": diabetes,
            "High Blood Pressure": high_bp,
            "Low HDL Cholesterol": low_hdl,
            "High LDL Cholesterol": high_ldl,
            "Alcohol Consumption": alcohol,
            "Stress Level": stress,
            "Sugar Consumption": sugar,
        }

        label, proba = predict(input_dict)

        if "High Risk" in label:
            st.error(f"Assessment Outcome: {label}")
        else:
            st.success(f"Assessment Outcome: {label}")

        st.caption(f"Model's predicted probability of Heart Disease: {proba:.1%}")


if __name__ == "__main__":
    main()
