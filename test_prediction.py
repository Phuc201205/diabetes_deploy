from predictor import (
    predict_diabetes,
    get_model_info
)


# ============================================================
# MODEL INFO
# ============================================================

info = get_model_info()


print(
    "===================================="
)

print(
    "MODEL INFO"
)

print(
    "===================================="
)


print(
    "Model:",
    info["model_name"]
)

print(
    "Version:",
    info["model_version"]
)

print(
    "Feature set:",
    info["feature_set"]
)

print(
    "Features:",
    info["num_features"]
)

print(
    "Threshold:",
    info["threshold"]
)


print(
    "\nDanh sách features:"
)


for i, feature in enumerate(
    info["features"],
    start=1
):

    print(
        f"{i:2}. {feature}"
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
# PREDICTION
# ============================================================

result = predict_diabetes(
    test_data
)


print(
    "\n===================================="
)

print(
    "PREDICTION RESULT"
)

print(
    "===================================="
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
    "Khuyến nghị:",
    result["recommendation"]
)


print(
    "\nMODEL INPUT:"
)

print(
    result["model_input"].T
)