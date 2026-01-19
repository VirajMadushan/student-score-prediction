"""
Student Exam Score Prediction Web Application
Using Ridge Regression (Baseline) and MLP Neural Network (Deep Learning)

Author: W.G.Viraj Madushan Jayaweera
Module: CIS6005 - Deep Learning
Competition: Kaggle Playground Series S6E1

How to run:
1. Install: pip install streamlit pandas numpy scikit-learn tensorflow
2. Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODELS AND PREPROCESSOR
# ============================================
@st.cache_resource
def load_models():
    """Load all saved models and metadata"""
    try:
        # Load Ridge model (full pipeline)
        with open('ridge_model.pkl', 'rb') as f:
            ridge_model = pickle.load(f)
        
        # Load MLP model
        mlp_model = keras.models.load_model('mlp_model.h5')
        
        # Load preprocessor
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Load feature info
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        
        # Load metrics
        with open('model_metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        
        return ridge_model, mlp_model, preprocessor, feature_info, metrics
    
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found! Make sure all .pkl and .h5 files are in the same folder.")
        st.error(f"Missing file: {e.filename}")
        st.stop()

# Load models
ridge_model, mlp_model, preprocessor, feature_info, metrics = load_models()

# ============================================
# MAIN TITLE AND HEADER
# ============================================
st.title("📚 Student Exam Score Prediction System")
st.markdown("""
This application predicts student exam scores using **Machine Learning** and **Deep Learning** models.  
Enter student information in the sidebar and click **Predict** to see results from both models.
""")

st.divider()

# ============================================
# SIDEBAR: USER INPUTS
# ============================================
st.sidebar.header("📝 Student Information")
st.sidebar.markdown("Enter the student's details below:")

# Numeric Inputs
st.sidebar.subheader("📊 Demographic & Academic")
age = st.sidebar.slider("Age", min_value=10, max_value=25, value=18, help="Student's age")
study_hours = st.sidebar.slider("Study Hours (per day)", min_value=0, max_value=20, value=5, help="Average daily study hours")
class_attendance = st.sidebar.slider("Class Attendance (%)", min_value=0, max_value=100, value=85, help="Percentage of classes attended")

st.sidebar.subheader("😴 Sleep & Lifestyle")
sleep_hours = st.sidebar.slider("Sleep Hours (per day)", min_value=4, max_value=12, value=7, help="Average daily sleep hours")
sleep_quality = st.sidebar.selectbox("Sleep Quality", options=["Poor", "Average", "Good"], index=2)

st.sidebar.subheader("🎓 Learning Environment")
internet_access = st.sidebar.selectbox("Internet Access", options=["No", "Yes"], index=1)
facility_rating = st.sidebar.slider("Facility Rating", min_value=1, max_value=10, value=7, help="School facility quality (1-10)")

st.sidebar.subheader("📖 Academic Details")
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
course = st.sidebar.selectbox("Course", options=["Math", "Science", "History", "English", "Art"])
study_method = st.sidebar.selectbox("Study Method", options=["Self-study", "Group study", "Coaching", "Online"])
exam_difficulty = st.sidebar.slider("Exam Difficulty", min_value=1, max_value=10, value=5, help="Expected exam difficulty (1-10)")

st.sidebar.divider()

# Predict Button
predict_button = st.sidebar.button("🎯 Predict Exam Score", type="primary", use_container_width=True)

# ============================================
# MAIN AREA: PREDICTIONS AND RESULTS
# ============================================

if predict_button:
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'study_hours': [study_hours],
        'class_attendance': [class_attendance],
        'sleep_hours': [sleep_hours],
        'exam_difficulty': [exam_difficulty],
        'facility_rating': [facility_rating],
        'gender': [gender],
        'course': [course],
        'study_method': [study_method],
        'internet_access': [internet_access],
        'sleep_quality': [sleep_quality]
    })
    
    # ============================================
    # PREDICTIONS
    # ============================================
    
    with st.spinner("🔮 Making predictions..."):
        
        # Ridge Regression Prediction (already includes preprocessing)
        ridge_pred = ridge_model.predict(input_data)[0]
        
        # MLP Prediction (needs manual preprocessing)
        input_preprocessed = preprocessor.transform(input_data)
        mlp_pred = mlp_model.predict(input_preprocessed, verbose=0)[0][0]
    
    # ============================================
    # DISPLAY RESULTS
    # ============================================
    
    st.success("✅ Predictions Complete!")
    
    # Create two columns for model comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 Ridge Regression (Baseline)")
        st.metric(
            label="Predicted Exam Score",
            value=f"{ridge_pred:.2f}",
            delta=None
        )
        st.caption(f"Model RMSE: {metrics['ridge_rmse']:.4f}")
        
        # Score interpretation
        if ridge_pred >= 80:
            st.success("🌟 Excellent Performance Expected!")
        elif ridge_pred >= 60:
            st.info("👍 Good Performance Expected")
        else:
            st.warning("⚠️ Needs Improvement")
    
    with col2:
        st.markdown("### 🟢 MLP Neural Network (Deep Learning)")
        st.metric(
            label="Predicted Exam Score",
            value=f"{mlp_pred:.2f}",
            delta=f"{mlp_pred - ridge_pred:+.2f}" if abs(mlp_pred - ridge_pred) > 0.5 else None
        )
        st.caption(f"Model RMSE: {metrics['mlp_rmse']:.4f}")
        
        # Score interpretation
        if mlp_pred >= 80:
            st.success("🌟 Excellent Performance Expected!")
        elif mlp_pred >= 60:
            st.info("👍 Good Performance Expected")
        else:
            st.warning("⚠️ Needs Improvement")
    
    st.divider()
    
    # ============================================
    # MODEL COMPARISON VISUALIZATION
    # ============================================
    
    st.markdown("### 📊 Model Comparison")
    
    col3, col4 = st.columns([2, 1])
    
    with col3:
        # Prediction comparison chart
        fig, ax = plt.subplots(figsize=(8, 4))
        models = ['Ridge\nRegression', 'MLP\nNeural Network']
        predictions = [ridge_pred, mlp_pred]
        colors = ['#3498db', '#2ecc71']
        
        bars = ax.bar(models, predictions, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Predicted Exam Score', fontsize=12, fontweight='bold')
        ax.set_title('Prediction Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, pred in zip(bars, predictions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pred:.2f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        st.pyplot(fig)
    
    with col4:
        st.markdown("#### 🎯 Model Performance")
        st.markdown(f"""
        **Ridge Regression:**
        - RMSE: `{metrics['ridge_rmse']:.4f}`
        - Type: Linear Model
        - Speed: ⚡ Fast
        
        **MLP Neural Network:**
        - RMSE: `{metrics['mlp_rmse']:.4f}`
        - Type: Deep Learning
        - Speed: 🐢 Slower
        """)
        
        # Determine better model
        if metrics['mlp_rmse'] < metrics['ridge_rmse']:
            st.success("✅ MLP performs better overall")
        else:
            st.info("✅ Ridge performs better overall")
    
    st.divider()
    
    # ============================================
    # KEY INSIGHTS
    # ============================================
    
    st.markdown("### 💡 Key Insights")
    
    insights = []
    
    if study_hours >= 8:
        insights.append("✅ **High study hours** strongly correlate with better performance")
    elif study_hours < 3:
        insights.append("⚠️ **Low study hours** may limit performance potential")
    
    if class_attendance >= 85:
        insights.append("✅ **Excellent attendance** is a strong performance indicator")
    elif class_attendance < 60:
        insights.append("⚠️ **Low attendance** significantly impacts learning outcomes")
    
    if sleep_quality == "Good" and sleep_hours >= 7:
        insights.append("✅ **Healthy sleep habits** support cognitive function")
    elif sleep_quality == "Poor" or sleep_hours < 6:
        insights.append("⚠️ **Poor sleep** may affect exam performance")
    
    if internet_access == "Yes":
        insights.append("✅ **Internet access** provides additional learning resources")
    
    if study_method in ["Coaching", "Group study"]:
        insights.append("✅ **Structured study method** enhances learning effectiveness")
    
    for insight in insights:
        st.markdown(insight)
    
    st.divider()
    
    # ============================================
    # INPUT SUMMARY
    # ============================================
    
    with st.expander("📋 View Input Summary"):
        st.dataframe(input_data, use_container_width=True)

else:
    # ============================================
    # WELCOME SCREEN (No prediction yet)
    # ============================================
    
    st.info("👈 Enter student information in the sidebar and click **Predict** to see results!")
    
    st.markdown("### 🎯 About This Application")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        **Features:**
        - 📊 Dual-model predictions (ML + DL)
        - 🎨 Interactive visualizations
        - 💡 Actionable insights
        - ⚡ Real-time predictions
        """)
    
    with col_b:
        st.markdown(f"""
        **Model Performance:**
        - Ridge RMSE: `{metrics['ridge_rmse']:.4f}`
        - MLP RMSE: `{metrics['mlp_rmse']:.4f}`
        - Dataset: 630,000 students
        """)
    
    st.markdown("### 📖 How It Works")
    st.markdown("""
    1. **Data Input**: Enter student demographics, study habits, and lifestyle factors
    2. **Preprocessing**: Features are automatically scaled and encoded
    3. **Dual Prediction**: Both Ridge Regression and MLP models generate predictions
    4. **Comparison**: See which model performs better for this specific student
    5. **Insights**: Get actionable recommendations based on the input data
    """)

# ============================================
# FOOTER
# ============================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>
    📚 Student Exam Score Prediction System | 
    CIS6005 Deep Learning Project | 
    Kaggle Playground Series S6E1
    </small>
</div>
""", unsafe_allow_html=True)