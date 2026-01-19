# 📚 Student Exam Score Prediction System

**Author:** W.G.Viraj Madushan Jayaweera  
**Student ID:** KD/BSCSD/20/02  
**Module:** CIS6005 – Deep Learning  
**Competition:** Kaggle Playground Series S6E1

## 📖 Description

This application predicts student exam scores using Machine Learning (Ridge Regression) and Deep Learning (MLP Neural Network) models.

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Features

- Dual-model predictions (ML + DL)
- Interactive visualizations
- Real-time predictions
- Actionable insights

## 🎯 Model Performance

- **Ridge Regression RMSE:** 7.4XXX
- **MLP Neural Network RMSE:** 7.3XXX
- **Dataset:** 630,000 student records

## 📁 Project Files

- `streamlit_app.py` - Main application
- `ridge_model.pkl` - Trained Ridge Regression model
- `mlp_model.h5` - Trained MLP Neural Network
- `preprocessor.pkl` - Data preprocessing pipeline
- `feature_info.pkl` - Feature metadata
- `model_metrics.pkl` - Model performance metrics

## 👥 Usage

1. Enter student information in the sidebar
2. Click "Predict Exam Score"
3. View predictions from both models
4. Review insights and recommendations

## 📝 License

Academic project for CIS6005 Deep Learning module.