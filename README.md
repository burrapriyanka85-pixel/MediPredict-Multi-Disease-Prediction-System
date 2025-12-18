# 🩺 MediPredict – Intelligent Multi-Disease Prediction System

## 📌 Project Overview
MediPredict is an intelligent machine learning–based healthcare application designed to predict the risk of multiple diseases using patient clinical data. The system integrates multiple trained machine learning models into a single Streamlit web interface, enabling early disease risk assessment in a simple and user-friendly manner.

This project is developed as a **Major Academic Project**, focusing on the practical application of machine learning techniques in the healthcare domain.

---

## 🎯 Diseases Covered
The system currently supports prediction for the following diseases:

- **Diabetes**
- **Heart Disease**
- **Liver Disease**
- **Parkinson’s Disease**

Each disease is handled using a **separate trained machine learning model**, ensuring better accuracy, modularity, and scalability.

---

## 🧠 Technologies Used
- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn  
- **Web Framework:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Model Persistence:** Pickle / Joblib  

---

## 🗂 Project Structure
MediPredict-Multi-Disease-Prediction-System/
│
├── data/ # Datasets used for training
│ ├── diabetes.csv
│ ├── Heart_Disease_Prediction.csv
│ ├── indian_liver_patient.csv
│ └── parkinsons.csv
│
├── frontend/ # Streamlit web application
│ └── app.py
│
├── models/ # Trained ML models (.pkl files)
│ ├── diabetes_model.pkl
│ ├── heart_model.pkl
│ ├── liver_model.pkl
│ └── parkinsons_model.pkl
│
├── utils/ # Prediction helper functions
│ └── prediction.py
│
├── train_diabetes.py # Diabetes model training script
├── train_heart.py # Heart disease model training script
├── train_liver.py # Liver disease model training script
├── train_parkinsons.py # Parkinson’s model training script
├── accuracy_plot.py # Model accuracy visualization
│
├── requirements.txt # Project dependencies
└── README.md

yaml
Copy code

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/burrapriyanka85-pixel/MediPredict-Multi-Disease-Prediction-System.git
cd MediPredict-Multi-Disease-Prediction-System
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the Streamlit application
bash
Copy code
streamlit run frontend/app.py
The application will open in your browser at:

arduino
Copy code
http://localhost:8501
📊 Machine Learning Workflow
Dataset collection and preprocessing

Feature selection and data cleaning

Model training and evaluation

Saving trained models as .pkl files

Integrating models with Streamlit UI for real-time prediction

🎓 Academic Relevance
Demonstrates an end-to-end machine learning pipeline

Applies ML concepts to real-world healthcare problems

Covers data preprocessing, model training, evaluation, and deployment

Suitable for major project submission and viva evaluation

⚠️ Disclaimer
This application is developed strictly for educational and research purposes only.
It is not intended for clinical diagnosis or medical decision-making.

👩‍💻 Author
Priyanka Burra
M.Sc Bioinformatics
