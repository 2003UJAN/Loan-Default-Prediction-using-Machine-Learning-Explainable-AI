import streamlit as st
import pandas as pd
import pickle
import shap

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Loan Default Prediction App")
st.write("Predict loan defaults using machine learning (Random Forest + SHAP explainability).")

# Sidebar for dataset upload
st.sidebar.header("Upload or Enter Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📂 Uploaded Data Preview:")
    st.write(df.head())
    if st.button("Predict on Uploaded Data"):
        predictions = model.predict(df)
        df["prediction"] = predictions
        st.write("🔮 Predictions:")
        st.write(df)
else:
    st.write("Or enter details manually:")

    loan_amount = st.number_input("Loan Amount", 1000, 50000, 10000)
    interest_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 10.0)
    income = st.number_input("Income", 20000, 200000, 50000)
    employment_years = st.slider("Employment Years", 0, 40, 5)
    credit_score = st.slider("Credit Score", 300, 850, 600)
    age = st.slider("Age", 18, 70, 30)
    existing_debt = st.number_input("Existing Debt", 0, 50000, 10000)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])

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
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        st.write(f"🔮 Default Prediction: **{'Yes (1)' if prediction == 1 else 'No (0)'}**")
        st.write(f"📊 Probability of Default: **{probability:.2f}**")

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        st.subheader("Explainability (SHAP values)")
        st.write("Feature impact on prediction:")
        st.bar_chart(pd.DataFrame(shap_values[1][0], index=input_data.columns, columns=["SHAP Value"]))
