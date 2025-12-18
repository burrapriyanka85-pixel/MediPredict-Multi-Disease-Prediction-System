# 🩺 MediPredict – Multi-Disease Prediction System

## Overview
MediPredict is a machine learning–based healthcare application that demonstrates an end-to-end ML workflow, from data preprocessing and model training to deployment through a Streamlit web interface. The project focuses on building reliable baseline machine learning models for predicting the risk of multiple diseases using structured clinical data.

The goal of this project is to showcase practical ML engineering skills, including data handling, model evaluation, modular code design, and user-facing deployment.

---

## Supported Disease Predictions
- Diabetes
- Heart Disease
- Liver Disease
- Parkinson’s Disease

Each disease is handled using an independently trained model, allowing modular development and easier future extension.

---

## Tech Stack
- **Language:** Python  
- **Machine Learning:** Scikit-learn  
- **Web Framework:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Model Serialization:** Pickle / Joblib  

---

## Project Structure
```text
MediPredict-Multi-Disease-Prediction-System/
├── data/
├── frontend/
│   └── app.py
├── models/
├── utils/
├── train_diabetes.py
├── train_heart.py
├── train_liver.py
├── train_parkinsons.py
├── accuracy_plot.py
├── requirements.txt
└── README.md
Model Performance
Baseline results obtained on held-out test data:

Disease	Model Used	Accuracy
Diabetes	Logistic Regression	~75%
Heart Disease	Logistic Regression	~92%
Liver Disease	Logistic Regression	~76%
Parkinson’s	Support Vector Machine	~82%

Model Selection Rationale
Logistic Regression was selected for diabetes, heart disease, and liver disease due to its simplicity, interpretability, and strong performance on structured clinical datasets.

Support Vector Machine (SVM) was chosen for Parkinson’s disease because it performs well on high-dimensional data with complex decision boundaries.

More complex models (e.g., Random Forest, XGBoost) were intentionally excluded to prioritize interpretability and faster experimentation.

Application Demo
(Add screenshots in a /screenshots folder)

text
Copy code
screenshots/home.png
screenshots/prediction.png
Running the Application Locally
1. Clone the repository
bash
Copy code
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
App runs at:

arduino
Copy code
http://localhost:8501
Future Improvements
Add explainability with SHAP or LIME

Deploy using Streamlit Cloud or HuggingFace Spaces

Introduce ensemble or gradient-boosting models

Expose predictions via a REST API

Disclaimer
This project is intended for demonstration and learning purposes only and should not be used for medical diagnosis.

Author
Priyanka Burra
Machine Learning | Healthcare Applications
