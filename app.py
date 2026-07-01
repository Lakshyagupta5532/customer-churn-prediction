import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Page config
st.set_page_config(page_title="ChurnShield - Customer Churn Predictor",
                   page_icon="🛡️", layout="wide")

# Custom CSS
st.markdown("""
<style>
.hero {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.hero p {
    font-size: 1.1rem;
    color: #888;
    margin-bottom: 0;
}
.section-header {
    font-size: 1rem;
    font-weight: 700;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}
.result-box {
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
}
.stButton button {
    border-radius: 8px;
    font-weight: 700;
    height: 3.2rem;
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <h1>🛡️ ChurnShield</h1>
    <p>AI-powered customer churn prediction — know who's leaving before they do.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Input Sections
st.markdown('<p class="section-header">Customer Details</p>', 
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("**👤 Personal Info**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12,
                          help="How long has this customer been with us?")

with col2:
    with st.container(border=True):
        st.markdown("**📱 Services**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines",
                                      ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service",
                                        ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security",
                                       ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup",
                                     ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection",
                                         ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support",
                                    ["Yes", "No", "No internet service"])

with col3:
    with st.container(border=True):
        st.markdown("**💳 Billing & Contract**")
        streaming_tv = st.selectbox("Streaming TV",
                                    ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies",
                                        ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract Type",
                                ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)",
                                           min_value=0.0, max_value=200.0,
                                           value=65.0)
        total_charges = st.number_input("Total Charges ($)",
                                         min_value=0.0, max_value=10000.0,
                                         value=1000.0)

st.write("")
predict_btn = st.button("🔮 Predict Churn Risk",
                         type="primary", use_container_width=True)

if predict_btn:
    input_dict = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
        'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'No' else 0,
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
        'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
        'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
        'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
        'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
        'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
        'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    churn_prob = probability[1]
    retain_prob = probability[0]

    st.divider()
    st.markdown('<p class="section-header">Prediction Result</p>',
                unsafe_allow_html=True)

    res_col1, res_col2, res_col3 = st.columns([1, 1, 1])

    with res_col1:
        if prediction == 1:
            st.error("### ⚠️ High Churn Risk")
            st.error("This customer is likely to **leave**. Consider retention strategy.")
        else:
            st.success("### ✅ Low Churn Risk")
            st.success("This customer is likely to **stay**. Keep up the good work!")

    with res_col2:
        st.metric("🚨 Churn Probability", f"{churn_prob:.1%}",
                  delta=f"{churn_prob - 0.5:.1%} vs baseline",
                  delta_color="inverse")
        st.progress(float(churn_prob))

    with res_col3:
        st.metric("💚 Retention Probability", f"{retain_prob:.1%}",
                  delta=f"{retain_prob - 0.5:.1%} vs baseline")
        st.progress(float(retain_prob))

    # Risk Level
    st.write("")
    if churn_prob >= 0.7:
        st.warning("🔴 **Critical Risk** — Immediate action recommended (discount, personal call)")
    elif churn_prob >= 0.4:
        st.warning("🟡 **Medium Risk** — Monitor closely, consider proactive outreach")
    else:
        st.info("🟢 **Low Risk** — Customer appears satisfied, maintain service quality")