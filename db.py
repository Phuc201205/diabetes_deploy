import pyodbc


# ============================================================
# CONFIG
# ============================================================

SERVER = r"LAPTOP-1I9RLDH2"
DATABASE = "DiabetesRiskDB"


# ============================================================
# CONNECTION
# ============================================================

def get_connection():
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )

    return pyodbc.connect(connection_string)


# ============================================================
# TEST CONNECTION
# ============================================================

def test_connection():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT DB_NAME()")

        database_name = cursor.fetchone()[0]

        print("Kết nối SQL Server thành công.")
        print("Database:", database_name)

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print("Kết nối thất bại:")
        print(e)

        return False


# ============================================================
# SAVE PREDICTION INPUT
# ============================================================

def save_prediction_input(data):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
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
    """

    cursor.execute(
        sql,
        data["age"],
        data["bmi"],
        data["general_health"],
        data["high_bp"],
        data["high_cholesterol"],
        data["race"],
        data["last_checkup"],
        data["drinks_alcohol"],
        data["sex"],
        data["has_personal_doctor"],
        data["employment_status"],
        data["had_heart_attack"]
    )

    case_id = cursor.fetchone()[0]

    conn.commit()

    cursor.close()
    conn.close()

    return case_id


# ============================================================
# SAVE PREDICTION RESULT
# ============================================================

def save_prediction_result(
    case_id,
    model_id,
    prediction_score,
    threshold,
    predicted_class,
    risk_level,
    recommendation
):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
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
    """

    cursor.execute(
        sql,
        case_id,
        model_id,
        prediction_score,
        threshold,
        predicted_class,
        risk_level,
        recommendation
    )

    conn.commit()

    cursor.close()
    conn.close()


# ============================================================
# SAVE COMPLETE PREDICTION
# ============================================================

def save_complete_prediction(
    input_data,
    model_id,
    prediction_score,
    threshold,
    predicted_class,
    risk_level,
    recommendation
):
    case_id = save_prediction_input(
        input_data
    )

    save_prediction_result(
        case_id=case_id,
        model_id=model_id,
        prediction_score=prediction_score,
        threshold=threshold,
        predicted_class=predicted_class,
        risk_level=risk_level,
        recommendation=recommendation
    )

    return case_id