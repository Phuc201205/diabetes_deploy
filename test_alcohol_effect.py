from copy import deepcopy

from predictor import predict_diabetes


# ============================================================
# HÀM SO SÁNH ALCOHOL NO / YES
# ============================================================

def compare_alcohol(profile_name, base_data):

    no_data = deepcopy(base_data)
    yes_data = deepcopy(base_data)

    no_data["drinks_alcohol"] = "No"
    yes_data["drinks_alcohol"] = "Yes"

    no_result = predict_diabetes(
        no_data
    )

    yes_result = predict_diabetes(
        yes_data
    )

    no_score = (
        no_result["prediction_score"]
        * 100
    )

    yes_score = (
        yes_result["prediction_score"]
        * 100
    )

    difference = (
        yes_score - no_score
    )

    print(
        "\n========================================"
    )

    print(
        profile_name
    )

    print(
        "========================================"
    )

    print(
        f"Alcohol = No : {no_score:.2f}%"
    )

    print(
        f"Alcohol = Yes: {yes_score:.2f}%"
    )

    print(
        f"Thay đổi     : {difference:+.2f} điểm %"
    )


    if difference < 0:

        print(
            "=> Alcohol = Yes làm score GIẢM"
        )

    elif difference > 0:

        print(
            "=> Alcohol = Yes làm score TĂNG"
        )

    else:

        print(
            "=> Alcohol không làm score thay đổi"
        )


# ============================================================
# PROFILE 1
# GẦN GIỐNG test_prediction.py HIỆN TẠI
# ============================================================

profile_1 = {

    "age": 55,

    "bmi": 31.5,

    "general_health": "Fair",

    "high_bp": "Yes",

    "high_cholesterol": "Yes",

    "race": "White",

    "last_checkup":
        "Within past year",

    "drinks_alcohol": "No",

    "sex": "Male",

    "has_personal_doctor":
        "Yes, only one",

    "employment_status":
        "Employed for wages",

    "had_heart_attack":
        "No"
}


# ============================================================
# PROFILE 2
# TRẺ HƠN - BMI BÌNH THƯỜNG
# ============================================================

profile_2 = {

    "age": 30,

    "bmi": 22.0,

    "general_health": "Very Good",

    "high_bp": "No",

    "high_cholesterol": "No",

    "race": "White",

    "last_checkup":
        "Within past year",

    "drinks_alcohol": "No",

    "sex": "Male",

    "has_personal_doctor":
        "Yes, only one",

    "employment_status":
        "Employed for wages",

    "had_heart_attack":
        "No"
}


# ============================================================
# PROFILE 3
# LỚN TUỔI - BMI CAO
# ============================================================

profile_3 = {

    "age": 70,

    "bmi": 35.0,

    "general_health": "Poor",

    "high_bp": "Yes",

    "high_cholesterol": "Yes",

    "race": "White",

    "last_checkup":
        "Within past year",

    "drinks_alcohol": "No",

    "sex": "Male",

    "has_personal_doctor":
        "Yes, only one",

    "employment_status":
        "Retired",

    "had_heart_attack":
        "No"
}


# ============================================================
# PROFILE 4
# TRUNG NIÊN - MỨC NGUY CƠ VỪA
# ============================================================

profile_4 = {

    "age": 45,

    "bmi": 27.0,

    "general_health": "Good",

    "high_bp": "No",

    "high_cholesterol": "Yes",

    "race": "White",

    "last_checkup":
        "Within past year",

    "drinks_alcohol": "No",

    "sex": "Female",

    "has_personal_doctor":
        "Yes, only one",

    "employment_status":
        "Employed for wages",

    "had_heart_attack":
        "No"
}


# ============================================================
# RUN
# ============================================================

compare_alcohol(
    "PROFILE 1 - Hồ sơ test hiện tại",
    profile_1
)

compare_alcohol(
    "PROFILE 2 - Trẻ, BMI bình thường",
    profile_2
)

compare_alcohol(
    "PROFILE 3 - Lớn tuổi, BMI cao",
    profile_3
)

compare_alcohol(
    "PROFILE 4 - Trung niên",
    profile_4
)