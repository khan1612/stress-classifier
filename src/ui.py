import pandas as pd
import streamlit as st

from src.config import PROBABILITY_LABELS, STRESS_COLORS, STRESS_LABELS


def configure_page(page_title):
    st.set_page_config(page_title=page_title, layout="wide")


def render_header(app_title):
    st.title(app_title)
    st.write(
        "Adjust the primary indicators below, or open the categorized sections "
        "for a more detailed analysis."
    )


def collect_user_inputs():
    inputs = {}

    left_col, right_col = st.columns(2)
    with left_col:
        inputs["age"] = st.slider("Age", 18, 60, 22)
        inputs["anxiety_tension"] = st.slider("Anxiety & Tension (0-10)", 0, 10, 5)
    with right_col:
        inputs["academic_overload"] = st.slider("Academic Overload (0-10)", 0, 10, 5)
        inputs["sleep_problems"] = st.slider("Sleep Problems (0-10)", 0, 10, 3)

    st.divider()
    _collect_physical_and_psychological_inputs(inputs)
    _collect_academic_and_social_inputs(inputs)

    return inputs


def _collect_physical_and_psychological_inputs(inputs):
    with st.expander("Physical & Psychological Indicators"):
        left_col, right_col = st.columns(2)
        with left_col:
            inputs["gender"] = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
            inputs["heartbeat_palpitations"] = st.slider("Heartbeat Palpitations", 0, 10, 2)
            inputs["headaches"] = st.slider("Headaches", 0, 10, 2)
            inputs["weight_changes"] = st.slider("Weight Changes", 0, 10, 1)
            inputs["health_issues"] = st.slider("General Health Issues", 0, 10, 2)
        with right_col:
            inputs["irritability"] = st.slider("Irritability", 0, 10, 3)
            inputs["restlessness"] = st.slider("Restlessness", 0, 10, 3)
            inputs["concentration_problems"] = st.slider("Concentration Problems", 0, 10, 3)
            inputs["sadness_low_mood"] = st.slider("Sadness/Low Mood", 0, 10, 2)
            inputs["stress_experience"] = st.slider("Overall Stress Experience", 0, 10, 5)


def _collect_academic_and_social_inputs(inputs):
    with st.expander("Academic & Social Environment"):
        left_col, right_col = st.columns(2)
        with left_col:
            inputs["peer_competition"] = st.slider("Peer Competition", 0, 10, 5)
            inputs["professor_difficulties"] = st.slider("Professor Difficulties", 0, 10, 3)
            inputs["academic_conflicts"] = st.slider("Academic Conflicts", 0, 10, 2)
            inputs["class_attendance"] = st.slider(
                "Class Attendance (0=Poor, 10=Perfect)", 0, 10, 8
            )
            inputs["low_academic_confidence"] = st.slider("Low Academic Confidence", 0, 10, 4)
        with right_col:
            inputs["subject_confidence"] = st.slider(
                "Subject Confidence (0=Low, 10=High)", 0, 10, 6
            )
            inputs["relationship_stress"] = st.slider("Relationship Stress", 0, 10, 3)
            inputs["loneliness_isolation"] = st.slider("Loneliness/Isolation", 0, 10, 2)
            inputs["work_environment"] = st.slider("Work/Study Environment Stress", 0, 10, 4)
            inputs["home_environment"] = st.slider("Home Environment Stress", 0, 10, 3)
            inputs["lack_relaxation_time"] = st.slider("Lack of Relaxation Time", 0, 10, 5)


def render_results(prediction, probabilities):
    st.divider()
    result_col, chart_col = st.columns(2)

    with result_col:
        label = STRESS_LABELS[prediction]
        color = STRESS_COLORS[prediction]

        st.subheader("Result")
        st.markdown(
            f"<h1 style='color:{color}'>{label}</h1>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{max(probabilities) * 100:.1f}%")

    with chart_col:
        st.subheader("Probability Distribution")
        probability_df = pd.DataFrame({"Probability": probabilities}, index=PROBABILITY_LABELS)
        st.bar_chart(probability_df)
