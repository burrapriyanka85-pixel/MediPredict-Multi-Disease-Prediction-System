# ==================================================
# IMPORTS
# ==================================================
import streamlit as st
import numpy as np
import joblib
import os
from streamlit_option_menu import option_menu

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="MediPredict | Intelligent Multi-Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# GLOBAL UI STYLES
# ==================================================
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
h1, h2, h3 {font-weight: 600;}
.stButton>button {
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# BASE DIRECTORY (PROJECT ROOT)
# ==================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================================================
# LOAD MODELS (PKL FILES)
# ==================================================
@st.cache_resource
def load_models():
    return {
        "diabetes": joblib.load(os.path.join(BASE_DIR, "models", "diabetes_model.pkl")),
        "heart": joblib.load(os.path.join(BASE_DIR, "models", "heart_model.pkl")),
        "liver": joblib.load(os.path.join(BASE_DIR, "models", "liver_model.pkl")),
        "parkinson": joblib.load(os.path.join(BASE_DIR, "models", "parkinsons_model.pkl")),
    }

MODELS = load_models()

# ==================================================
# RESULT DISPLAY (CORPORATE STYLE)
# ==================================================
def show_result(prediction, disease, name):
    st.markdown("## Clinical Assessment Outcome")
    if prediction == 1:
        st.error(f"⚠️ Elevated risk indicators detected for {name}")
        st.markdown(f"**Condition:** {disease}")
        st.markdown("**Recommendation:** Clinical evaluation advised")
    else:
        st.success(f"✅ No significant risk indicators detected for {name}")
        st.markdown(f"**Condition:** {disease}")
        st.markdown("**Recommendation:** Routine monitoring")

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.title("🩺 MediPredict")
    st.caption("AI-Powered Clinical Risk Intelligence")

    selected = option_menu(
        "Prediction Modules",
        ["Diabetes", "Heart Disease", "Liver Disease", "Parkinson’s"],
        icons=["activity", "heart", "droplet", "person"],
        default_index=0
    )

    st.markdown("---")
    st.caption("Decision-Support Tool • Not a Medical Diagnosis")

# ==================================================
# DIABETES
# ==================================================
if selected == "Diabetes":
    st.title("Diabetes Risk Assessment")
    name = st.text_input("Patient Name")

    c1, c2, c3 = st.columns(3)
    preg = c1.number_input("Pregnancies", 0, 20)
    glucose = c2.number_input("Glucose Level", 0, 300)
    bp = c3.number_input("Blood Pressure", 0, 200)
    skin = c1.number_input("Skin Thickness", 0, 100)
    insulin = c2.number_input("Insulin", 0, 900)
    bmi = c3.number_input("BMI", 0.0, 70.0)
    dpf = c1.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = c2.number_input("Age", 1, 120)

    if st.button("Run Diabetes Risk Analysis"):
        pred = MODELS["diabetes"].predict(
            [[preg, glucose, bp, skin, insulin, bmi, dpf, age]]
        )[0]
        show_result(pred, "Diabetes", name)

# ==================================================
# HEART DISEASE
# ==================================================
elif selected == "Heart Disease":
    st.title("Heart Disease Risk Assessment")
    name = st.text_input("Patient Name")

    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 1, 120)
    sex = c2.selectbox("Gender", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0
    cp = c3.number_input("Chest Pain Type (0–3)", 0, 3)
    trestbps = c1.number_input("Resting Blood Pressure", 80, 200)
    chol = c2.number_input("Cholesterol", 100, 400)
    fbs = c3.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = c1.number_input("Rest ECG (0–2)", 0, 2)
    thalach = c2.number_input("Max Heart Rate", 60, 220)
    exang = c3.selectbox("Exercise Angina", [0, 1])
    oldpeak = c1.number_input("ST Depression", 0.0, 6.0)
    slope = c2.number_input("Slope (0–2)", 0, 2)
    ca = c3.number_input("Major Vessels (0–3)", 0, 3)
    thal = c1.number_input("Thal (0–3)", 0, 3)

    if st.button("Run Heart Disease Risk Analysis"):
        pred = MODELS["heart"].predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal]]
        )[0]
        show_result(pred, "Heart Disease", name)

# ==================================================
# LIVER DISEASE
# ==================================================
elif selected == "Liver Disease":
    st.title("Liver Disease Risk Assessment")
    name = st.text_input("Patient Name")

    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 1, 100)
    gender = c2.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0
    tb = c3.number_input("Total Bilirubin", 0.0, 10.0)
    db = c1.number_input("Direct Bilirubin", 0.0, 5.0)
    alk = c2.number_input("Alkaline Phosphatase", 50, 300)
    alt = c3.number_input("ALT", 10, 300)
    ast = c1.number_input("AST", 10, 300)
    tp = c2.number_input("Total Proteins", 4.0, 9.0)
    alb = c3.number_input("Albumin", 2.0, 6.0)
    ag = c1.number_input("Albumin/Globulin Ratio", 0.5, 2.5)

    if st.button("Run Liver Disease Risk Analysis"):
        pred = MODELS["liver"].predict(
            [[gender, age, tb, db, alk, alt, ast, tp, alb, ag]]
        )[0]
        show_result(pred, "Liver Disease", name)

# ==================================================
# PARKINSON’S (CORPORATE – 5 FEATURES ONLY)
# ==================================================
elif selected == "Parkinson’s":
    st.title("Parkinson’s Disease Risk Assessment")
    st.caption("AI-assisted voice biomarker screening")

    name = st.text_input("Patient Name")

    col1, col2 = st.columns(2)

    fo = col1.number_input("Average Vocal Frequency (Fo Hz)", 50.0, 300.0)
    jitter = col1.number_input("Jitter (%)", 0.0, 1.0)
    shimmer = col2.number_input("Shimmer", 0.0, 1.0)
    nhr = col2.number_input("Noise-to-Harmonics Ratio (NHR)", 0.0, 1.0)
    hnr = col1.number_input("Harmonics-to-Noise Ratio (HNR)", 0.0, 40.0)

    if st.button("Run Parkinson’s Risk Analysis"):
        features = [[fo, jitter, shimmer, nhr, hnr]]
        pred = MODELS["parkinson"].predict(features)[0]
        show_result(pred, "Parkinson’s Disease", name)
