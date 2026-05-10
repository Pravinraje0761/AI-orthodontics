# =========================
# STEP 13: SAVE THE MODEL
# =========================

import pickle
from pathlib import Path

import pandas as pd


def save_model(model: object, model_path: str = "gender_rf_model.pkl") -> None:
    with Path(model_path).open("wb") as f:
        pickle.dump(model, f)


# print("Model saved successfully as gender_rf_model.pkl")
# =========================
# STEP 14: BASIC DEPLOYMENT FUNCTION
# =========================

def predict_gender(
    gonial_angle,
    condylar_height,
    coronoid_ramus_height,
    intercondylar_distance,
    sigmoid_notch_angle,
    mental_foramen_to_inferior_border_ramus,
    model_path="gender_rf_model.pkl"
):
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    new_data = pd.DataFrame([{
        'Gonial_Angle': gonial_angle,
        'Condylar_Height': condylar_height,
        'Coronoid_Ramus_Height': coronoid_ramus_height,
        'Intercondylar_Distance': intercondylar_distance,
        'Sigmoid_Notch_Angle': sigmoid_notch_angle,
        'Mental_Foramen_to_Inferior_Border_Ramus': mental_foramen_to_inferior_border_ramus
    }])

    prediction = loaded_model.predict(new_data)[0]
    probability = loaded_model.predict_proba(new_data)[0]

    gender_label = "Male" if prediction == 1 else "Female"

    return {
        "Predicted_Gender": gender_label,
        "Probability_Female": probability[0],
        "Probability_Male": probability[1]
    }

if __name__ == "__main__":
    print("Use save_model(...) and predict_gender(...) from this module.")