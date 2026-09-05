import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

PREPROCESSOR_PATH = (
    "models/brfss_preprocessor.joblib"
)

MODEL_PATH = (
    "models/brfss_xgboost_model.json"
)

METADATA_PATH = (
    "models/diabetes_model_metadata.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

model = xgb.XGBClassifier()

model.load_model(
    MODEL_PATH
)


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


FEATURES = metadata["features"]

THRESHOLD = float(
    metadata["threshold"]
)


# ============================================================
# AGE → BRFSS AGE CODE
# ============================================================

def age_to_code(age):

    if 18 <= age <= 24:
        return 1.0

    elif 25 <= age <= 29:
        return 2.0

    elif 30 <= age <= 34:
        return 3.0

    elif 35 <= age <= 39:
        return 4.0

    elif 40 <= age <= 44:
        return 5.0

    elif 45 <= age <= 49:
        return 6.0

    elif 50 <= age <= 54:
        return 7.0

    elif 55 <= age <= 59:
        return 8.0

    elif 60 <= age <= 64:
        return 9.0

    elif 65 <= age <= 69:
        return 10.0

    elif 70 <= age <= 74:
        return 11.0

    elif 75 <= age <= 79:
        return 12.0

    elif age >= 80:
        return 13.0

    else:
        raise ValueError(
            "Model BRFSS này chỉ áp dụng cho người từ 18 tuổi trở lên."
        )


# ============================================================
# GENERAL HEALTH
# ============================================================

def general_health_to_code(value):

    mapping = {
        "Excellent": 1.0,
        "Very Good": 2.0,
        "Good": 3.0,
        "Fair": 4.0,
        "Poor": 5.0
    }

    return mapping[value]


# ============================================================
# LAST CHECKUP
# ============================================================

def checkup_to_code(value):

    mapping = {
        "Within past year": 1.0,
        "Within past 2 years": 2.0,
        "Within past 5 years": 3.0,
        "5 or more years ago": 4.0,
        "Never": 5.0
    }

    return mapping[value]


# ============================================================
# YES / NO / UNKNOWN → BOOLEAN
# ============================================================

def yes_no_unknown(value):

    if value == "Yes":
        return True

    elif value == "No":
        return False

    elif value == "Unknown":
        return np.nan

    raise ValueError(
        f"Giá trị không hợp lệ: {value}"
    )


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_model_input(data):

    row = {
        "gen_health_00":
            general_health_to_code(
                data["general_health"]
            ),

        "bmi_00":
            float(
                data["bmi"]
            ),

        "age_00":
            age_to_code(
                int(data["age"])
            ),

        "high_bp_00":
            data["high_bp"],

        "high_cholesterol_00":
            yes_no_unknown(
                data["high_cholesterol"]
            ),

        "race_00":
            data["race"],

        "l_checkup_00":
            checkup_to_code(
                data["last_checkup"]
            ),

        "drinks_alcohol_00":
            yes_no_unknown(
                data["drinks_alcohol"]
            ),

        "sex_00":
            data["sex"],

        "has_personal_doctor_00":
            data["has_personal_doctor"],

        "employment_status_00":
            data["employment_status"],

        "had_heart_attack_00":
            yes_no_unknown(
                data["had_heart_attack"]
            )
    }


    df = pd.DataFrame(
        [row],
        columns=FEATURES
    )

    return df


# ============================================================
# PREDICT
# ============================================================

def predict_diabetes(data):

    raw_input = create_model_input(
        data
    )


    processed_input = (
        preprocessor.transform(
            raw_input
        )
    )


    prediction_score = float(
        model.predict_proba(
            processed_input
        )[0, 1]
    )


    predicted_class = int(
        prediction_score
        >= THRESHOLD
    )


    if predicted_class == 1:

        screening_result = (
            "Vượt ngưỡng sàng lọc"
        )

        recommendation = (
            "Điểm sàng lọc của mô hình vượt ngưỡng. "
            "Bạn nên cân nhắc thực hiện xét nghiệm "
            "đường huyết hoặc HbA1c và trao đổi "
            "với nhân viên y tế."
        )

    else:

        screening_result = (
            "Không vượt ngưỡng sàng lọc"
        )

        recommendation = (
            "Điểm sàng lọc của mô hình chưa vượt ngưỡng. "
            "Kết quả này không loại trừ hoàn toàn nguy cơ "
            "tiểu đường và không thay thế xét nghiệm y khoa."
        )


    return {
        "prediction_score":
            prediction_score,

        "threshold":
            THRESHOLD,

        "predicted_class":
            predicted_class,

        "screening_result":
            screening_result,

        "recommendation":
            recommendation,

        "model_input":
            raw_input
    }