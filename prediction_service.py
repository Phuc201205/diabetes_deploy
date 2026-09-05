from db import get_connection
from predictor import predict_diabetes


# ============================================================
# LẤY MODEL ID
# ============================================================

def get_model_id(
    model_version="BRFSS-2023-v1"
):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT model_id
            FROM ModelVersion
            WHERE model_version = ?
            """,
            model_version
        )

        row = cursor.fetchone()

        if row is None:
            raise ValueError(
                f"Không tìm thấy model_version: "
                f"{model_version}"
            )

        return int(row[0])

    finally:
        cursor.close()
        conn.close()


# ============================================================
# PREDICT + SAVE SQL
# ============================================================

def predict_and_save(input_data):

    # --------------------------------------------------------
    # 1. MODEL PREDICTION
    # --------------------------------------------------------

    result = predict_diabetes(
        input_data
    )

    model_id = get_model_id()


    # --------------------------------------------------------
    # 2. DATABASE CONNECTION
    # --------------------------------------------------------

    conn = get_connection()
    cursor = conn.cursor()

    try:

        # ====================================================
        # 3. INSERT PredictionInput
        # ====================================================

        cursor.execute(
            """
            INSERT INTO PredictionInput
            (
                age,
                bmi,
                general_health,
                high_bp,
                high_cholesterol,
                race,
                last_checkup,
                drinks_alcohol,
                sex,
                has_personal_doctor,
                employment_status,
                had_heart_attack
            )
            OUTPUT INSERTED.case_id
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,

            input_data["age"],
            input_data["bmi"],
            input_data["general_health"],
            input_data["high_bp"],
            input_data["high_cholesterol"],
            input_data["race"],
            input_data["last_checkup"],
            input_data["drinks_alcohol"],
            input_data["sex"],
            input_data["has_personal_doctor"],
            input_data["employment_status"],
            input_data["had_heart_attack"]
        )

        case_id = int(
            cursor.fetchone()[0]
        )


        # ====================================================
        # 4. INSERT PredictionResult
        # ====================================================

        cursor.execute(
            """
            INSERT INTO PredictionResult
            (
                case_id,
                model_id,
                prediction_score,
                threshold,
                predicted_class,
                risk_level,
                recommendation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,

            case_id,
            model_id,
            result["prediction_score"],
            result["threshold"],
            result["predicted_class"],
            result["screening_result"],
            result["recommendation"]
        )


        # ====================================================
        # 5. COMMIT
        # ====================================================

        conn.commit()


        return {
            **result,
            "case_id": case_id,
            "model_id": model_id
        }


    except Exception:

        conn.rollback()

        raise


    finally:

        cursor.close()
        conn.close()