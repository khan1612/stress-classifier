import pandas as pd


def predict_stress_level(model, scaler, feature_names, user_inputs):
    input_df = pd.DataFrame([user_inputs])[feature_names]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    return prediction, probabilities
