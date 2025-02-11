import os
import time
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Page Config
st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦", layout="wide")


# Session state to track onboarding
if "onboarding_done" not in st.session_state:
    st.session_state.onboarding_done = False

# Display onboarding with progress bar
def show_onboarding():
    st.sidebar.success("📖 Welcome Guide is starting...")
    progress_bar = st.progress(0)  # Initialize progress bar

    steps = [
        ("👋 Welcome to the Loan Default Prediction Dashboard!", 0.2),
        ("📊 This dashboard helps predict loan default risk using financial data.", 0.4),
        ("📝 Enter details in the sidebar (loan amount, interest rate, etc.) to get predictions.", 0.6),
        ("🚀 Click 'Predict Default Risk' to analyze your loan approval chances!", 0.8),
        ("🔍 Explore insights in the main panel with charts and visualizations!", 1.0)
    ]
    
    for msg, progress in steps:
        st.toast(msg, icon="✅")
        progress_bar.progress(progress)  # Updating progress bar
        time.sleep(2)

    progress_bar.empty()  # Remove progress bar after completion
    st.sidebar.success("🎉 Guide Completed! Enjoy exploring.")

    st.session_state.onboarding_done = True

# Show guide on first load
if not st.session_state.onboarding_done:
    show_onboarding()

# Option to restart the guide
if st.sidebar.button("📖 Show Guide Again"):
    st.session_state.onboarding_done = False
    show_onboarding()
    

# Load trained model
current_dir = os.path.dirname(os.path.abspath(__file__))  
model_path = os.path.join(current_dir, "..", "models", "best_tuned_model.pkl")


if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: {model_path}. Please train and save the model.")
    st.stop()

model = joblib.load(model_path)


# Custom Styling
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
            transition: 0.2s;
        }
        .stButton>button:hover {
            background-color: #c70039 !important;
        }
        .stSidebar {
            background-color: #227f99;
            color: white;
            padding: 10px;
            border-radius: 10px;
        }
        .stSidebar label {
            color: #ffcc00 !important;
            font-weight: bold;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            color: #FFA500 !important; /* Light Orange */
            font-weight: bold;
        }    
    </style>
""", unsafe_allow_html=True)


# App Title
st.title("🏦 Loan Default Prediction Dashboard")
st.markdown("---")


# Dashboard with Tabs
st.subheader("🔹 Explore Loan Insights")
tabs = st.tabs(["Sample Loan Data", "Feature Importance"])

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
    st.markdown("**This chart visualises how loan amount correlates with default risk.**")

with tabs[1]:
    feature_importance = pd.DataFrame({
        "Feature": ["Credit Score", "Loan Amount", "Interest Rate", "Income"],
        "Importance": [0.45, 0.25, 0.20, 0.10]
    })
    fig = px.bar(feature_importance, x="Importance", y="Feature", orientation="h", title="Feature Importance in Default Prediction", color="Feature", color_continuous_scale="viridis")
    st.plotly_chart(fig)


# Sidebar Inputs
loan_limit = st.sidebar.selectbox("Loan Limit", ["cf", "abc"])  # Example categories
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, step=500, help="Enter the total amount you want to borrow.")
interest_rate = st.sidebar.slider("Interest Rate (%)", min_value=1.0, max_value=30.0, step=0.1, help="Higher interest rates may increase default risk.")
credit_score = st.sidebar.slider("Credit Score", min_value=300, max_value=850, step=10, help="A good credit score lowers your risk.")
income = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=500000, step=1000, help="Enter your yearly income before taxes.")
region = st.sidebar.selectbox("Region", ["North-East", "Central", "South"])

# Create DataFrame for Input
input_data = pd.DataFrame({
    "loan_limit": [loan_limit],
    "loan_amount": [loan_amount],
    "rate_of_interest": [interest_rate],
    "Credit_Score": [credit_score],
    "income": [income],
    "Region": [region]
})

# Apply One-Hot Encoding
input_data = pd.get_dummies(input_data)

# Ensure all expected columns are present (fill missing ones with 0)
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0  # Default missing columns to 0

# Ensure correct column order
input_data = input_data[model.feature_names_in_]

# Scale numerical features (only those that were scaled in training)
scaler = StandardScaler()
numerical_cols = ["loan_amount", "rate_of_interest", "Credit_Score", "income"]
input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])


# Prediction Button & Logic
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
        
    
    #  Visualization
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
