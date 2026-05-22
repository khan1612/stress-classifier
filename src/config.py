from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "stress_classifier_final.pkl"

PAGE_TITLE = "Comprehensive Stress Classifier"
APP_TITLE = "Student Stress Level Classifier"

STRESS_LABELS = {
    0: "High Stress",
    1: "Medium Stress",
    2: "Low Stress",
}

STRESS_COLORS = {
    0: "#d93636",
    1: "#cc7a00",
    2: "#008f5a",
}

PROBABILITY_LABELS = ["High", "Medium", "Low"]
