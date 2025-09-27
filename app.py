import streamlit as st
import pandas as pd
import pickle
import shap
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Model Setup -----
MODEL_URL = "https://huggingface.co/spaces/ujan2003/loan-default-prediction/resolve/main/model.pkl"
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model..."):
        r = requests.get(MODEL_URL)
        open(MODEL_PATH, "wb").write(r.content)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ----- Page Config -----
st.set_page_config(
    page_title="💳 Loan Default Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Header -----
st.markdown("""
<div style='text-align: center; background-color: #f0f2f6; padding: 20px; border-radius: 15px'>
<h1 style='color:#1f77b4;'>💳 Loan Default Prediction Dashboard</h1>
<p>Predict loan defaults in real-time with explainable AI</p>
</div>
""", unsafe_allow_html=True)

# ----- Sidebar -----
st.sidebar.header("Upload or Enter Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
st.sidebar.markdown("OR enter data manually in the main panel.")

# ----- Tabs -----
tab1, tab2 = st.tabs(["📂 Upload CSV", "✏️ Manual Input"])

# ----- Tab 1: CSV Upload -----
with tab1:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview:")
        st.dataframe(df.head())

        if st.button("Predict Defaults for Uploaded Data"):
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)[:,1]
            df["Default Prediction"] = predictions
            df["Default Probability"] = probabilities
            st.write("🔮 Predictions with Probability:")
            
            # Color-coded probability
            def color_prob(val):
                if val < 0.3:
                    color = 'green'
                elif val < 0.7:
                    color = 'orange'
                else:
                    color = 'red'
                return f'color: {color}; font-weight: bold'
            
            st.dataframe(df.style.applymap(color_prob, subset=["Default Probability"]))
            st.success("✅ Predictions completed!")

# ----- Tab 2: Manual Input -----
with tab2:
    st.subheader("Enter Loan Applicant Details")
    col1, col2 = st.columns(2)
    
    with col1:
        loan_amount = st.number_input("Loan Amount", 1000, 50000, 10000, step=1000, help="Total loan amount requested")
        income = st.number_input("Income", 20000, 200000, 50000, step=5000, help="Applicant annual income")
        credit_score = st.slider("Credit Score", 300, 850, 600, help="Credit score from 300-850")
        existing_debt = st.number_input("Existing Debt", 0, 50000, 10000, step=500, help="Existing outstanding debts")
    
    with col2:
        interest_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 10.0, help="Interest rate of the loan")
        employment_years = st.slider("Employment Years", 0, 40, 5, help="Years of employment")
        age = st.slider("Age", 18, 70, 30, help="Applicant age")
        loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], help="Loan tenure in months")
    
    input_data = pd.DataFrame([{
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "income": income,
        "employment_years": employment_years,
        "credit_score": credit_score,
        "age": age,
        "existing_debt": existing_debt,
        "loan_term": loan_term
    }])

    if st.button("Predict Loan Default"):
        with st.spinner("Predicting..."):
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

        # Color-coded probability
        if probability < 0.3:
            prob_color = 'green'
        elif probability < 0.7:
            prob_color = 'orange'
        else:
            prob_color = 'red'

        # Metrics display
        col1, col2 = st.columns(2)
        col1.metric("🔮 Default Prediction", "Yes (1)" if prediction == 1 else "No (0)")
        col2.markdown(f"📊 Probability of Default: <span style='color:{prob_color}; font-weight:bold'>{probability:.2f}</span>", unsafe_allow_html=True)

        # ----- SHAP Explainability -----
        st.subheader("📈 Feature Impact (SHAP Values)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # Handle binary classification
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_for_default = shap_values[1][0]  # Class 1 = default
        else:
            shap_for_default = shap_values[0]  # fallback

        shap_df = pd.DataFrame(shap_for_default, index=input_data.columns, columns=["SHAP Value"])
        shap_df = shap_df.sort_values("SHAP Value", ascending=True)

        # Plot
        plt.figure(figsize=(8,5))
        sns.barplot(x="SHAP Value", y=shap_df.index, data=shap_df, palette="coolwarm")
        plt.title("Feature Importance Impact")
        st.pyplot(plt)

# ----- Footer -----
st.markdown("""
<div style='text-align: center; padding: 10px; margin-top: 30px; color: #888; font-size: 14px;'>
Developed by Ujan Pradhan | Powered by Streamlit & Hugging Face
</div>
""", unsafe_allow_html=True)
