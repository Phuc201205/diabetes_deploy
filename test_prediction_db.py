from prediction_service import (
    predict_and_save
)


# ============================================================
# TEST DATA
# ============================================================

test_data = {

    "age":
        55,

    "bmi":
        31.5,

    "general_health":
        "Fair",

    "high_bp":
        "Yes",

    "high_cholesterol":
        "Yes",

    "race":
        "White",

    "last_checkup":
        "Within past year",

    "sex":
        "Male",

    "has_personal_doctor":
        "Yes, only one",

    "employment_status":
        "Employed for wages",

    "had_heart_attack":
        "No"
}


# ============================================================
# PREDICT + SAVE
# ============================================================

result = predict_and_save(
    test_data
)


print(
    "===================================="
)

print(
    "PREDICTION + SQL"
)

print(
    "===================================="
)


print(
    "Case ID:",
    result["case_id"]
)

print(
    "Model ID:",
    result["model_id"]
)

print(
    "Model Version:",
    result["model_version"]
)

print(
    "Score:",
    result["prediction_score"]
)

print(
    "Threshold:",
    result["threshold"]
)

print(
    "Class:",
    result["predicted_class"]
)

print(
    "Kết quả:",
    result["screening_result"]
)


print(
    "\nĐã lưu vào SQL Server thành công."
)