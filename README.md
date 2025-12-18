# 🩺 MediPredict – Multi-Disease Prediction System

## Overview
MediPredict is a machine learning–based healthcare application that demonstrates an end-to-end ML workflow, from data preprocessing and model training to deployment through a Streamlit web interface.  
The project focuses on building reliable **baseline machine learning models** for predicting the risk of multiple diseases using structured clinical data.

The goal of this project is to showcase practical ML engineering skills, including data handling, model evaluation, modular code design, and user-facing deployment.

---

## Supported Disease Predictions
The application currently supports prediction for the following conditions:

- Diabetes  
- Heart Disease  
- Liver Disease  
- Parkinson’s Disease  

Each disease is handled using an **independently trained model**, allowing modular development and easier future extension.

---

## Tech Stack
- **Language:** Python  
- **Machine Learning:** Scikit-learn  
- **Web Framework:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Model Serialization:** Pickle / Joblib  

---

## Project Structure
MediPredict-Multi-Disease-Prediction-System/
│
├── data/ # Clinical datasets
├── frontend/ # Streamlit application
│ └── app.py
├── models/ # Trained ML models
├── utils/ # Prediction helper logic
│
├── train_diabetes.py
├── train_heart.py
├── train_liver.py
├── train_parkinsons.py
├── accuracy_plot.py
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## Model Performance
The following results were obtained using held-out test data after training and evaluation.  
These metrics are intended to reflect **baseline model performance**, not optimized or production-tuned results.

| Disease        | Model Used             | Accuracy |
|---------------|------------------------|----------|
| Diabetes      | Logistic Regression    | ~75%     |
| Heart Disease | Logistic Regression    | ~92%     |
| Liver Disease | Logistic Regression    | ~76%     |
| Parkinson’s   | Support Vector Machine | ~82%     |

---

## Model Selection Rationale
- **Logistic Regression** was chosen for diabetes, heart disease, and liver disease prediction due to its simplicity, interpretability, and strong performance on structured clinical datasets. It serves as a reliable baseline and allows easier understanding of feature influence.

- **Support Vector Machine (SVM)** was selected for Parkinson’s disease prediction because the dataset exhibits complex feature relationships where SVMs perform well in high-dimensional spaces and non-linear decision boundaries.

More complex models such as Random Forest or XGBoost were intentionally excluded in this version to prioritize interpretability, faster experimentation, and clarity of results.

---

## Application Demo
Screenshots below show the working Streamlit interface.

### Home Interface
![Home Screen](screenshots/home.png)

### Prediction Output
![Prediction Result](screenshots/prediction.png)

---

## Running the Application Locally

### 1. Clone the repository
```bash
git clone https://github.com/burrapriyanka85-pixel/MediPredict-Multi-Disease-Prediction-System.git
cd MediPredict-Multi-Disease-Prediction-System
2. Install dependencies
bash
Copy code
pip install -r requirements.txt
3. Launch the application
bash
Copy code
streamlit run frontend/app.py
The app will be available at:

arduino
Copy code
http://localhost:8501
Future Improvements
Add model explainability using SHAP or LIME

Deploy the application using Streamlit Cloud or HuggingFace Spaces

Introduce ensemble or gradient-boosting models

Expose prediction functionality through a REST API

Disclaimer
This project is intended for educational and demonstration purposes only and should not be used as a substitute for professional medical advice or diagnosis.

Author
Priyanka Burra
M.Sc Bioinformatics
Machine Learning & Healthcare Applications
