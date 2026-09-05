from db import save_complete_prediction


test_input = {
    "age": 55,
    "bmi": 31.5,
    "general_health": "Fair",
    "high_bp": "Yes",
    "high_cholesterol": "Yes",
    "race": "White",
    "last_checkup": "Within past year",
    "drinks_alcohol": "No",
    "sex": "Male",
    "has_personal_doctor": "Yes",
    "employment_status": "Employed",
    "had_heart_attack": "No"
}


prediction_score = 0.72
threshold = 0.481958
predicted_class = 1
risk_level = "High"

recommendation = (
    "Điểm sàng lọc ở mức cao. "
    "Nên cân nhắc thực hiện xét nghiệm "
    "đường huyết hoặc HbA1c tại cơ sở y tế."
)


case_id = save_complete_prediction(
    input_data=test_input,
    model_id=1,
    prediction_score=prediction_score,
    threshold=threshold,
    predicted_class=predicted_class,
    risk_level=risk_level,
    recommendation=recommendation
)


print("Lưu prediction thành công.")
print("case_id:", case_id)