import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="Comprehensive Stress Classifier", layout="wide")

@st.cache_resource
def load_resources():
    package = joblib.load('stress_classifier_final.pkl')
    return package['model'], package['scaler'], package['features']

model, scaler, feature_names = load_resources()

st.title("Student Stress Level Classifier ")
st.write("Adjust the primary indicators below, or open the categorized sections for a more detailed analysis.")

# --- INPUT COLLECTION ---
inputs = {}

# 1. Primary Indicators (Always visible)
col_a, col_b = st.columns(2)
with col_a:
    inputs['age'] = st.slider('Age', 18, 60, 22)
    inputs['anxiety_tension'] = st.slider('Anxiety & Tension (0-10)', 0, 10, 5)
with col_b:
    inputs['academic_overload'] = st.slider('Academic Overload (0-10)', 0, 10, 5)
    inputs['sleep_problems'] = st.slider('Sleep Problems (0-10)', 0, 10, 3)

st.divider()

# 2. Detailed Indicators (In Expanders to save space)
with st.expander("Physical & Psychological Indicators"):
    c1, c2 = st.columns(2)
    with c1:
        inputs['gender'] = st.selectbox('Gender (0=Female, 1=Male)', [0, 1])
        inputs['heartbeat_palpitations'] = st.slider('Heartbeat Palpitations', 0, 10, 2)
        inputs['headaches'] = st.slider('Headaches', 0, 10, 2)
        inputs['weight_changes'] = st.slider('Weight Changes', 0, 10, 1)
        inputs['health_issues'] = st.slider('General Health Issues', 0, 10, 2)
    with c2:
        inputs['irritability'] = st.slider('Irritability', 0, 10, 3)
        inputs['restlessness'] = st.slider('Restlessness', 0, 10, 3)
        inputs['concentration_problems'] = st.slider('Concentration Problems', 0, 10, 3)
        inputs['sadness_low_mood'] = st.slider('Sadness/Low Mood', 0, 10, 2)
        inputs['stress_experience'] = st.slider('Overall Stress Experience', 0, 10, 5)

with st.expander("Academic & Social Environment"):
    c3, c4 = st.columns(2)
    with c3:
        inputs['peer_competition'] = st.slider('Peer Competition', 0, 10, 5)
        inputs['professor_difficulties'] = st.slider('Professor Difficulties', 0, 10, 3)
        inputs['academic_conflicts'] = st.slider('Academic Conflicts', 0, 10, 2)
        inputs['class_attendance'] = st.slider('Class Attendance (0=Poor, 10=Perfect)', 0, 10, 8)
        inputs['low_academic_confidence'] = st.slider('Low Academic Confidence', 0, 10, 4)
    with c4:
        inputs['subject_confidence'] = st.slider('Subject Confidence (0=Low, 10=High)', 0, 10, 6)
        inputs['relationship_stress'] = st.slider('Relationship Stress', 0, 10, 3)
        inputs['loneliness_isolation'] = st.slider('Loneliness/Isolation', 0, 10, 2)
        inputs['work_environment'] = st.slider('Work/Study Environment Stress', 0, 10, 4)
        inputs['home_environment'] = st.slider('Home Environment Stress', 0, 10, 3)
        inputs['lack_relaxation_time'] = st.slider('Lack of Relaxation Time', 0, 10, 5)

# --- PROCESSING ---
# Create DataFrame ensuring columns match training data exactly
input_df = pd.DataFrame([inputs])[feature_names]
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)[0]
probs = model.predict_proba(input_scaled)[0]
labels = {0: "High Stress", 1: "Medium Stress", 2: "Low Stress"}
colors = {0: "#FF4B4B", 1: "#FFA500", 2: "#00CC96"}

# --- RESULTS ---
st.divider()
res_col1, res_col2 = st.columns(2)

with res_col1:
    st.subheader("Result")
    st.markdown(f"<h1 style='color:{colors[prediction]}'>{labels[prediction]}</h1>", unsafe_allow_html=True)
    st.metric("Confidence", f"{max(probs)*100:.1f}%")

with res_col2:
    st.subheader("Probability Distribution")
    st.bar_chart(pd.DataFrame({'Prob': probs}, index=['High', 'Medium', 'Low']))