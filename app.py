
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model + columns
# -----------------------------
model = joblib.load("churn_model.pkl")
cols = joblib.load("model_columns.pkl")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Customer Churn Prediction")

st.markdown("Enter customer details below to predict churn probability.")
st.markdown("---")

# -----------------------------
# INPUT LAYOUT
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.number_input("Tenure", min_value=0, max_value=100, value=1)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=100.0)

with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    phone = st.selectbox("Phone Service", ["No", "Yes"])

# -----------------------------
# Internet / Contract / Billing
# -----------------------------
st.markdown("### Service & Billing")

col3, col4 = st.columns(2)

with col3:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

with col4:
    online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    payment = st.selectbox(
        "Payment Method",
        ["Credit card (automatic)", "Bank transfer (automatic)", "Electronic check", "Mailed check"]
    )

# -----------------------------
# Convert inputs → model format
# -----------------------------
input_dict = {
    "SeniorCitizen": SeniorCitizen,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,

    # One-hot encoded categorical features
    "gender_Male": 1 if gender == "Male" else 0,
    "Partner_Yes": 1 if partner == "Yes" else 0,
    "Dependents_Yes": 1 if dependents == "Yes" else 0,
    "PhoneService_Yes": 1 if phone == "Yes" else 0,

    "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
    "InternetService_No": 1 if internet == "No" else 0,

    "OnlineSecurity_Yes": 1 if online_sec == "Yes" else 0,
    "OnlineSecurity_No internet service": 1 if online_sec == "No internet service" else 0,

    "DeviceProtection_Yes": 1 if device_prot == "Yes" else 0,
    "DeviceProtection_No internet service": 1 if device_prot == "No internet service" else 0,

    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,

    "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0,

    "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
    "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
}

# Many columns like "_No internet service" or "_No phone service" are already handled by fill_value=0 below.

# Build DataFrame & align with model columns
df = pd.DataFrame([input_dict]).reindex(columns=cols, fill_value=0)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.error(f"⚠️ Customer Likely to Churn\n**Probability: {prob:.2f}**")
    else:
        st.success(f"🙂 Customer Likely to Stay\n**Probability: {prob:.2f}**")
