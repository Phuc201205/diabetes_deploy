from predictor import predict_diabetes


test_data = {
    "age": 55,

    "bmi": 31.5,

    "general_health": "Fair",

    "high_bp": "Yes",

    "high_cholesterol": "Yes",

    "race": "White",

    "last_checkup": "Within past year",

    "drinks_alcohol": "No",

    "sex": "Male",

    "has_personal_doctor":
        "Yes, only one",

    "employment_status":
        "Employed for wages",

    "had_heart_attack":
        "No"
}


result = predict_diabetes(
    test_data
)


print(
    "===================================="
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
    result["model_input"]
)