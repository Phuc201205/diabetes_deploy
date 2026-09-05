from db import (
    test_connection,
    get_connection
)


# ============================================================
# 1. TEST CONNECTION
# ============================================================

test_connection()


# ============================================================
# 2. READ MODEL VERSION
# ============================================================

try:
    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            model_id,
            model_name,
            model_version,
            algorithm,
            threshold,
            recall,
            precision_score,
            f2_score,
            average_precision,
            roc_auc
        FROM ModelVersion
        """
    )

    rows = cursor.fetchall()

    print(
        "\nMODEL VERSION:"
    )

    for row in rows:
        print(row)

    cursor.close()
    conn.close()

except Exception as e:
    print(
        "\nKhông đọc được ModelVersion:"
    )

    print(e)