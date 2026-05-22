from src.config import APP_TITLE, PAGE_TITLE
from src.model_loader import load_model_package
from src.prediction import predict_stress_level
from src.ui import collect_user_inputs, configure_page, render_header, render_results


def main():
    configure_page(PAGE_TITLE)

    model, scaler, feature_names = load_model_package()
    render_header(APP_TITLE)

    user_inputs = collect_user_inputs()
    prediction, probabilities = predict_stress_level(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        user_inputs=user_inputs,
    )

    render_results(prediction, probabilities)
