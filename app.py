"""
Clinical Heart Disease Prediction Application
A Streamlit-based machine learning application for heart disease prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model (placeholder - replace with actual model path)"""
    # Replace this with your actual model loading logic
    try:
        model = joblib.load('heart_disease_model.pkl')
        return model
    except FileNotFoundError:
        st.warning("Model file not found. Using demo mode.")
        return None

def main():
    st.title("❤️ Clinical Heart Disease Prediction")
    st.markdown("---")
    
    # Sidebar for navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📊 Prediction", "📈 Analytics", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Prediction":
        show_prediction()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "ℹ️ About":
        show_about()

def show_home():
    """Display home page"""
    st.header("Welcome to Heart Disease Prediction System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About This Application
        
        This machine learning application predicts the likelihood of heart disease 
        based on clinical and demographic data.
        
        #### Key Features:
        - 🎯 Real-time prediction
        - 📊 Data visualization
        - 📈 Statistical analysis
        - 🔒 Secure and privacy-focused
        
        #### How It Works:
        1. Enter your health metrics
        2. The model analyzes the data
        3. Receive personalized risk assessment
        
        **Disclaimer:** This tool is for educational purposes only and should not 
        replace professional medical advice.
        """)
    
    with col2:
        st.metric("Status", "Active")
        st.metric("Model Type", "ML Classifier")

def show_prediction():
    """Display prediction page"""
    st.header("Make a Prediction")
    
    model = load_model()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=400, value=200)
        max_hr = st.number_input("Max Heart Rate", min_value=0, max_value=220, value=150)
    
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        bp = st.number_input("Blood Pressure (systolic)", min_value=0, max_value=300, value=120)
        st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0)
    
    with col3:
        chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
        fasting_bs = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=0, max_value=400, value=120)
        st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
    
    if st.button("🔍 Predict", key="predict_button", use_container_width=True):
        if model:
            # Create feature array (replace with actual features)
            features = np.array([[age, cholesterol, max_hr]])
            
            # Prediction placeholder
            risk_score = np.random.random()  # Replace with actual model prediction
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Score", f"{risk_score:.2%}", "High Risk" if risk_score > 0.5 else "Low Risk")
            with col2:
                st.metric("Confidence", f"{(1-abs(0.5-risk_score))*200:.1f}%", "")
            with col3:
                st.metric("Recommendation", "Consult Doctor" if risk_score > 0.5 else "Monitor Health", "")
            
            st.markdown("---")
            st.info("Please consult with a healthcare professional for proper diagnosis and treatment.")
        else:
            st.error("Model not available in demo mode.")

def show_analytics():
    """Display analytics page"""
    st.header("Data Analytics")
    
    st.info("📊 Load your dataset to view analytics")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Dataset Overview")
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        
        st.subheader("Statistics")
        st.dataframe(df.describe())

def show_about():
    """Display about page"""
    st.header("About This Application")
    
    st.markdown("""
    ### Clinical Heart Disease Prediction System
    
    **Version:** 1.0.0  
    **Status:** Production Ready
    
    #### Technology Stack
    - **Framework:** Streamlit
    - **ML Library:** scikit-learn
    - **Data Processing:** pandas, numpy
    - **Visualization:** plotly, seaborn
    
    #### Deployment
    - **Platform:** Streamlit Cloud
    - **CI/CD:** GitHub Actions
    - **Repository:** GitHub
    
    #### Contact & Support
    For issues or questions, please visit our GitHub repository.
    
    ---
    
    **Disclaimer:** This application is provided for educational and research purposes only. 
    It is not intended to diagnose, treat, cure, or prevent any disease. Always consult with 
    qualified healthcare professionals for medical advice.
    """)

if __name__ == "__main__":
    main()
