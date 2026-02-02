# 📚 Student Exam Score Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**An intelligent web application that predicts student exam performance using Machine Learning and Deep Learning**

[Live Demo](https://viraj-student-predictor.streamlit.app/) 

</div>

---

## 🎯 Overview

This project leverages both traditional **Machine Learning (Ridge Regression)** and modern **Deep Learning (Multi-Layer Perceptron)** to predict student exam scores based on demographic, behavioral, and educational features. Built with Streamlit, it provides an intuitive web interface for real-time predictions and insights.

### ✨ Key Features

- 🤖 **Dual-Model Predictions**: Compare Ridge Regression baseline with MLP Neural Network
- 📊 **Interactive Visualizations**: Real-time charts and model performance comparisons
- 💡 **Actionable Insights**: AI-powered recommendations based on student data
- ⚡ **Fast Performance**: Optimized models with sub-second prediction times
- 🎨 **Modern UI**: Clean, responsive design built with Streamlit

---

## 📸 Screenshots

<div align="center">

### Main Interface
![image alt](https://github.com/VirajMadushan/student-score-prediction/blob/60860c97efb7dd81c1f1636e67a11b963f975610/Main%20Interface.png)

</div>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/VirajMadushan/student-score-predictor.git
cd student-score-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run streamlit_app.py
```

4. **Open your browser**
```
Navigate to http://localhost:8501
```

---

## 📊 Model Performance

| Model | RMSE | Architecture | Training Time |
|-------|------|--------------|---------------|
| **Ridge Regression** | 7.4XXX | Linear Model | ~5 seconds |
| **MLP Neural Network** | 7.3XXX | 128→64→1 Dense Layers | ~2 minutes |

**Dataset**: 630,000 student records from Kaggle Playground Series S6E1

---

## 🏗️ Project Structure

```
student-score-predictor/
│
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── models/                    # Trained models (download separately)
│   ├── ridge_model.pkl
│   ├── mlp_model.h5
│   ├── preprocessor.pkl
│   ├── feature_info.pkl
│   └── model_metrics.pkl
│
├── notebooks/
│   └── training_pipeline.ipynb  # Model training notebook
│
└── screenshots/               # UI screenshots
    ├── main_interface.png
    ├── predictions.png
    └── model_comparison.png
```

---

## 🧠 How It Works

### 1. Data Input
Users enter student information through an interactive sidebar:
- Demographics (age, gender)
- Study habits (hours, method, attendance)
- Lifestyle factors (sleep quality, internet access)
- Academic details (course, exam difficulty)

### 2. Preprocessing Pipeline
```python
Numeric Features → Median Imputation → Standard Scaling
Categorical Features → Mode Imputation → One-Hot Encoding
```

### 3. Model Prediction
- **Ridge Regression**: Linear baseline model for interpretability
- **MLP Neural Network**: Deep learning model for capturing non-linear patterns

### 4. Results & Insights
- Side-by-side model comparison
- Performance interpretation
- Actionable recommendations

---

## 🔧 Technical Stack

### Frontend
- **Streamlit** - Interactive web framework
- **Matplotlib & Seaborn** - Data visualization

### Backend
- **Scikit-learn** - Machine learning pipeline
- **TensorFlow/Keras** - Deep learning framework
- **Pandas & NumPy** - Data processing

### Machine Learning
- **Ridge Regression** - Regularized linear model
- **MLP** - Multi-layer perceptron with dropout regularization

---

## 📈 Model Architecture

### Ridge Regression Pipeline
```
Input Features → Preprocessing → Ridge(alpha=1.0) → Prediction
```

### MLP Neural Network
```
Input Layer (n features)
    ↓
Dense Layer (128 units, ReLU)
    ↓
Dropout (30%)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (30%)
    ↓
Output Layer (1 unit)
```

**Training Configuration:**
- Optimizer: Adam
- Loss: Mean Squared Error
- Early Stopping: Patience = 3 epochs
- Batch Size: 256

---

## 📝 Input Features

### Numeric Features (6)
- `age` - Student's age
- `study_hours` - Daily study hours
- `class_attendance` - Attendance percentage
- `sleep_hours` - Daily sleep hours
- `exam_difficulty` - Exam difficulty rating (1-10)
- `facility_rating` - School facility quality (1-10)

### Categorical Features (5)
- `gender` - Male/Female
- `course` - Math, Science, History, English, Art
- `study_method` - Self-study, Group study, Coaching, Online
- `internet_access` - Yes/No
- `sleep_quality` - Poor, Average, Good

---

## 🎓 Academic Context

**Project Details:**
- **Course**: CIS6005 - Deep Learning
- **Institution**: Cardiff met
- **Student**: W.G. Viraj Madushan Jayaweera
- **Student ID**: KD/BSCSD/20/02
- **Competition**: Kaggle Playground Series S6E1

**Learning Outcomes:**
- Applied end-to-end machine learning pipeline
- Compared traditional ML with deep learning approaches
- Deployed production-ready web application
- Practiced model evaluation and comparison

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Correlation analysis of numeric features
- Distribution analysis of target variable
- Categorical feature impact assessment

### 2. Data Preprocessing
- Missing value imputation
- Feature scaling and normalization
- One-hot encoding for categorical variables

### 3. Model Development
- **Baseline**: Ridge Regression with L2 regularization
- **Advanced**: MLP with dropout and early stopping

### 4. Evaluation
- Train-validation split (80-20)
- RMSE as primary metric
- Cross-model comparison

---

## 📦 Requirements

```txt
streamlit==1.31.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.7.2
tensorflow==2.20.0
matplotlib==3.7.1
seaborn==0.12.2
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🐛 Known Issues

- Model files are large (>100MB) - use Git LFS for version control
- First prediction takes ~2 seconds due to model loading
- Requires stable internet for Streamlit Cloud deployment

---

## 🔮 Future Enhancements

- [ ] Add batch prediction via CSV upload
- [ ] Implement SHAP values for model explainability
- [ ] Add confidence intervals for predictions
- [ ] Include feature importance visualization
- [ ] Develop mobile-responsive design
- [ ] Add user authentication and history tracking
- [ ] Implement A/B testing for model comparison

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Kaggle** - For providing the Playground Series S6E1 dataset
- **Anthropic** - For Claude AI assistance in development
- **Streamlit** - For the amazing web framework
- **TensorFlow Team** - For the deep learning framework
- **My Instructors** - For guidance and support

---

## 📚 References

1. [Kaggle Playground Series S6E1](https://www.kaggle.com/competitions/playground-series-s6e1)
2. [Streamlit Documentation](https://docs.streamlit.io)
3. [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
4. [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by Viraj Jayaweera

</div>
