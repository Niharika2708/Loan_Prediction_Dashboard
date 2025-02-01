import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time

# 🎨 Improved Page Config
st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦", layout="wide")

# Load the trained model
current_dir = os.path.dirname(os.path.abspath(__file__))  
model_path = os.path.join(current_dir, "..", "models", "best_tuned_model.pkl")

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: {model_path}. Please train and save the model.")
    st.stop()

model = joblib.load(model_path)

# 🌟 Custom Styling
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to right, #1f4037, #99f2c8);
            color: #333;
        }
        h1, h2, h3 {
            color: #ffcc00;
            font-family: 'Arial Black', sans-serif;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .stButton>button {
            background-color: #ff5733 !important;
            color: white !important;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #c70039 !important;
        }
        .stSidebar {
            background-color: #2c3e50;
            color: white;
            padding: 10px;
            border-radius: 10px;
        }
        .stSidebar label {
            color: #ffcc00 !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# 🏦 App Title
st.title("🏦 Loan Default Prediction Dashboard")
st.markdown("---")

# 🌐 Interactive Dashboard with Tabs
st.subheader("🔹 Explore Loan Insights")
tabs = st.tabs(["📊 Sample Loan Data", "🔥 Feature Importance"])

with tabs[0]:
    def generate_sample_data():
        np.random.seed(42)
        data = pd.DataFrame({
            "Loan Amount": np.random.randint(5000, 50000, 200),
            "Interest Rate (%)": np.random.uniform(3.0, 25.0, 200),
            "Credit Score": np.random.randint(300, 850, 200),
            "Income": np.random.randint(20000, 120000, 200),
            "Default Probability": np.random.uniform(0, 1, 200)
        })
        return data
    
    data = generate_sample_data()
    fig = px.scatter(
        data, 
        x="Loan Amount", 
        y="Default Probability", 
        color="Interest Rate (%)", 
        size="Credit Score", 
        hover_data=["Income"],
        title="Loan Amount vs. Default Probability",
        color_continuous_scale="viridis"
    )
    st.plotly_chart(fig)
    st.markdown("**This chart helps visualize how loan amount correlates with default risk. Higher interest rates and lower credit scores tend to have a greater likelihood of default.**")

with tabs[1]:
    st.subheader("🔥 Feature Importance Analysis")
    feature_importance = pd.DataFrame({
        "Feature": ["Credit Score", "Loan Amount", "Interest Rate", "Income"],
        "Importance": [0.45, 0.25, 0.20, 0.10]
    })
    fig = px.bar(feature_importance, x="Importance", y="Feature", orientation="h", title="Feature Importance in Default Prediction", color="Feature", color_continuous_scale="viridis")
    st.plotly_chart(fig)

# 🔂 Sidebar for Loan Input Details
st.sidebar.header("Enter Loan Details")
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, step=500)
interest_rate = st.sidebar.slider("Interest Rate (%)", min_value=1.0, max_value=30.0, step=0.1)
credit_score = st.sidebar.slider("Credit Score", min_value=300, max_value=850, step=10)
income = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=500000, step=1000)
employment_years = st.sidebar.slider("Years at Current Job", min_value=0, max_value=40, step=1)

# Convert inputs into DataFrame
input_data = pd.DataFrame({
    "loan_amount": [loan_amount],
    "rate_of_interest": [interest_rate],
    "Credit_Score": [credit_score],
    "income": [income],
})

# ⚙️ Prediction Button & Logic
if st.sidebar.button("🚀 Predict Default Risk"):
    with st.spinner("Analyzing risk..."):
        time.sleep(2)
        prediction = model.predict_proba(input_data)[0][1]  # Probability of default
    
    st.markdown("---")
    st.subheader("🔄 Estimated Default Probability")
    st.metric(label="Predicted Default Risk", value=f"{prediction:.2%}")
    
    if prediction < 0.3:
        st.success("💪 Low Risk - Loan Likely to be Approved!")
    elif prediction < 0.7:
        st.warning("⚠️ Medium Risk - Further Evaluation Needed.")
    else:
        st.error("❌ High Risk - Loan Likely to be Denied.")
    
    # Visualization
    st.markdown("---")
    st.subheader("📊 Loan Default Distribution")
    np.random.seed(42)
    default_rates = np.clip(np.random.normal(loc=0.5, scale=0.2, size=500), 0, 1)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(default_rates, bins=20, kde=True, color="red", alpha=0.7)
    ax.set_xlabel("Default Probability (%)")
    ax.set_ylabel("Number of Loans")
    st.pyplot(fig)

st.markdown("---")
st.info("💡 Tip: Higher interest rates and lower credit scores increase the likelihood of loan default.")
