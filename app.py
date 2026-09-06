from textwrap import dedent

import pandas as pd
import streamlit as st

from predictor import preprocessor
from prediction_service import predict_and_save
from db import get_connection


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Diabetes Risk Screening",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# HELPER RENDER HTML
# ============================================================

def render_html(content):
    """
    Render HTML trực tiếp bằng Streamlit.
    Không đưa HTML qua Markdown parser.
    """

    st.html(
        dedent(content)
    )


# ============================================================
# STYLE
# ============================================================

render_html(
    """
    <style>

    /* ========================================================
       TOÀN TRANG
    ======================================================== */

    .stApp {
        background:
            linear-gradient(
                180deg,
                #f4f9ff 0%,
                #ffffff 38%,
                #f8fafc 100%
            );
    }


    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ========================================================
       HERO
    ======================================================== */

    .hero-box {
        padding: 1.8rem 2rem;
        margin-bottom: 1.2rem;

        background:
            linear-gradient(
                135deg,
                #eff6ff,
                #f8fbff
            );

        border: 1px solid #dbeafe;
        border-radius: 20px;

        box-shadow:
            0 6px 24px
            rgba(15, 23, 42, 0.05);
    }


    .hero-label {
        font-size: 0.82rem;
        font-weight: 700;

        color: #2563eb;

        letter-spacing: 0.08em;
        text-transform: uppercase;

        margin-bottom: 0.5rem;
    }


    .hero-title {
        font-size: 2.15rem;
        font-weight: 750;

        color: #0f172a;

        margin-bottom: 0.55rem;
    }


    .hero-description {
        max-width: 850px;

        color: #64748b;

        font-size: 1rem;
        line-height: 1.7;
    }


    /* ========================================================
       SECTION
    ======================================================== */

    .section-box {
        padding: 1rem 1.2rem;

        background: #ffffff;

        border: 1px solid #e2e8f0;
        border-radius: 14px;

        margin-top: 1.2rem;
        margin-bottom: 1rem;
    }


    .section-title {
        font-size: 1.1rem;
        font-weight: 700;

        color: #0f172a;

        margin-bottom: 0.2rem;
    }


    .section-description {
        color: #64748b;

        font-size: 0.9rem;
        line-height: 1.5;
    }


    /* ========================================================
       INPUT
    ======================================================== */

    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }


    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
    }


    /* ========================================================
       BUTTON
    ======================================================== */

    div[data-testid="stFormSubmitButton"] button {
        height: 48px;

        border-radius: 10px;

        font-size: 1rem;
        font-weight: 700;
    }


    /* ========================================================
       METRIC
    ======================================================== */

    div[data-testid="stMetric"] {
        padding: 1rem 1.2rem;

        background: white;

        border: 1px solid #e2e8f0;
        border-radius: 14px;

        box-shadow:
            0 3px 12px
            rgba(15, 23, 42, 0.04);
    }


    /* ========================================================
       RESULT TITLE
    ======================================================== */

    .result-title {
        margin-top: 1rem;
        margin-bottom: 0.8rem;

        font-size: 1.35rem;
        font-weight: 700;

        color: #0f172a;
    }


    /* ========================================================
       RISK SCORE CARD
    ======================================================== */

    .risk-score-card {
        padding: 1.5rem 1.6rem;

        border-radius: 16px;
        border: 1px solid;

        margin-top: 1rem;
        margin-bottom: 1rem;

        box-shadow:
            0 4px 18px
            rgba(15, 23, 42, 0.04);
    }


    .risk-score-safe {
        background: #f0fdf4;
        border-color: #bbf7d0;
    }


    .risk-score-alert {
        background: #fef2f2;
        border-color: #fecaca;
    }


    .risk-score-label {
        color: #64748b;

        font-size: 0.82rem;
        font-weight: 700;

        letter-spacing: 0.07em;
        text-transform: uppercase;

        margin-bottom: 0.4rem;
    }


    .risk-score-value {
        font-size: 2.6rem;
        font-weight: 800;

        line-height: 1.1;

        margin-bottom: 0.2rem;
    }


    .risk-score-safe .risk-score-value {
        color: #16a34a;
    }


    .risk-score-alert .risk-score-value {
        color: #dc2626;
    }


    .risk-status {
        margin-top: 0.35rem;

        font-size: 1rem;
        font-weight: 700;
    }


    .risk-score-safe .risk-status {
        color: #15803d;
    }


    .risk-score-alert .risk-status {
        color: #b91c1c;
    }


    /* ========================================================
       SCORE BAR
    ======================================================== */

    .risk-bar {
        width: 100%;
        height: 12px;

        background: #e2e8f0;

        border-radius: 999px;

        overflow: hidden;

        margin-top: 1rem;
        margin-bottom: 1rem;
    }


    .risk-bar-fill-safe {
        height: 100%;

        background: #22c55e;

        border-radius: 999px;
    }


    .risk-bar-fill-alert {
        height: 100%;

        background: #ef4444;

        border-radius: 999px;
    }


    .risk-score-detail {
        color: #475569;

        font-size: 0.95rem;
        line-height: 1.75;
    }


    .risk-score-note {
        margin-top: 0.8rem;

        color: #64748b;

        font-size: 0.85rem;
        line-height: 1.6;
    }


    /* ========================================================
       NOTE
    ======================================================== */

    .medical-note {
        margin-top: 1rem;

        color: #64748b;

        font-size: 0.9rem;
        line-height: 1.7;

        padding: 1rem 1.2rem;

        background: #f8fafc;

        border: 1px solid #e2e8f0;
        border-radius: 12px;
    }

    </style>
    """
)


# ============================================================
# HELPER
# LẤY CATEGORY TỪ FITTED PREPROCESSOR
# ============================================================

def get_fitted_categories(column_name):
    """
    Lấy category chính xác mà OneHotEncoder
    đã học khi huấn luyện.
    """

    for _, transformer, columns in preprocessor.transformers_:

        try:
            column_list = list(columns)

        except TypeError:
            continue


        if column_name not in column_list:
            continue


        encoder = None


        # Nếu transformer là sklearn Pipeline
        if hasattr(
            transformer,
            "named_steps"
        ):

            for step in (
                transformer
                .named_steps
                .values()
            ):

                if hasattr(
                    step,
                    "categories_"
                ):

                    encoder = step
                    break


        # Nếu transformer chính là OneHotEncoder
        elif hasattr(
            transformer,
            "categories_"
        ):

            encoder = transformer


        if encoder is None:
            continue


        column_index = (
            column_list.index(
                column_name
            )
        )


        values = (
            encoder.categories_[
                column_index
            ]
        )


        clean_values = []


        for value in values:

            try:

                is_missing = pd.isna(
                    value
                )

            except Exception:

                is_missing = False


            if is_missing:
                continue


            if hasattr(
                value,
                "item"
            ):

                value = value.item()


            clean_values.append(
                value
            )


        return clean_values


    return []


def require_categories(column_name):
    """
    Không tự đoán category.

    Nếu không đọc được category từ fitted model,
    ứng dụng sẽ dừng.
    """

    values = get_fitted_categories(
        column_name
    )


    if not values:

        st.error(
            "Không đọc được category của "
            f"{column_name} từ preprocessor."
        )

        st.stop()


    return values


def get_default_index(
    options,
    preferred_value
):
    """
    Trả về index của giá trị mặc định.

    Nếu không tồn tại thì dùng phần tử đầu tiên.
    """

    try:

        return list(options).index(
            preferred_value
        )

    except ValueError:

        return 0


# ============================================================
# TRANSLATION LABELS
# ============================================================

GENERAL_HEALTH_LABELS = {

    "Excellent":
        "Xuất sắc",

    "Very Good":
        "Rất tốt",

    "Good":
        "Tốt",

    "Fair":
        "Trung bình",

    "Poor":
        "Kém"
}


CHECKUP_LABELS = {

    "Within past year":
        "Trong vòng 1 năm qua",

    "Within past 2 years":
        "Từ 1 đến dưới 2 năm",

    "Within past 5 years":
        "Từ 2 đến dưới 5 năm",

    "5 or more years ago":
        "Từ 5 năm trở lên",

    "Never":
        "Chưa bao giờ"
}


YES_NO_UNKNOWN_LABELS = {

    "No":
        "Không",

    "Yes":
        "Có",

    "Unknown":
        "Không biết / chưa kiểm tra"
}


BP_LABELS = {

    "No":
        "Không",

    "Yes":
        "Có",

    "Borderline":
        "Mức ranh giới"
}


SEX_LABELS = {

    "Male":
        "Nam",

    "Female":
        "Nữ"
}


DOCTOR_LABELS = {

    "No":
        "Không",

    "Yes, only one":
        "Có, một bác sĩ / nơi chăm sóc chính",

    "More than one":
        "Có, nhiều hơn một"
}


RACE_LABELS = {

    "White":
        "Da trắng",

    "Black":
        "Da đen / Người Mỹ gốc Phi",

    "Asian":
        "Người châu Á",

    "Hispanic":
        "Hispanic / Latino",

    "Multiracial":
        "Đa chủng tộc",

    "American Indian or Alaskan Native":
        "Người bản địa Mỹ / Alaska",

    "Native Hawaiian or other Pacific Islander":
        "Người Hawaii bản địa / Đảo Thái Bình Dương",

    "Other":
        "Khác"
}


EMPLOYMENT_LABELS = {

    "Employed for wages":
        "Làm công ăn lương",

    "Self-employed":
        "Tự kinh doanh",

    "Out of work for 1 year or more":
        "Thất nghiệp từ 1 năm trở lên",

    "Out of work for less than 1 year":
        "Thất nghiệp dưới 1 năm",

    "A homemaker":
        "Nội trợ",

    "A Homemaker":
        "Nội trợ",

    "A student":
        "Học sinh / Sinh viên",

    "A Student":
        "Học sinh / Sinh viên",

    "Retired":
        "Đã nghỉ hưu",

    "Unable to work":
        "Không có khả năng làm việc"
}


# ============================================================
# MODEL CATEGORIES
# ============================================================

BP_OPTIONS = require_categories(
    "high_bp_00"
)


RACE_OPTIONS = require_categories(
    "race_00"
)


SEX_OPTIONS = require_categories(
    "sex_00"
)


DOCTOR_OPTIONS = require_categories(
    "has_personal_doctor_00"
)


EMPLOYMENT_OPTIONS = require_categories(
    "employment_status_00"
)


# ============================================================
# DATABASE HISTORY
# ============================================================

def load_prediction_history(
    limit=50
):

    limit = max(
        1,
        min(
            int(limit),
            100
        )
    )


    conn = get_connection()

    cursor = conn.cursor()


    try:

        cursor.execute(
            f"""
            SELECT TOP {limit}

                pr.prediction_id,
                pr.case_id,

                pi.age,
                pi.bmi,
                pi.sex,
                pi.general_health,

                pr.prediction_score,
                pr.threshold,
                pr.predicted_class,
                pr.risk_level,

                mv.model_version,

                pr.created_at

            FROM PredictionResult AS pr

            INNER JOIN PredictionInput AS pi
                ON pr.case_id = pi.case_id

            INNER JOIN ModelVersion AS mv
                ON pr.model_id = mv.model_id

            ORDER BY
                pr.created_at DESC
            """
        )


        rows = cursor.fetchall()


        columns = [
            item[0]
            for item
            in cursor.description
        ]


        data = [
            tuple(row)
            for row in rows
        ]


        return pd.DataFrame(
            data,
            columns=columns
        )


    finally:

        cursor.close()

        conn.close()


# ============================================================
# HEADER
# ============================================================

render_html(
    """
    <div class="hero-box">

        <div class="hero-label">
            Diabetes Risk Screening
        </div>

        <div class="hero-title">
            🩺 Hệ thống sàng lọc nguy cơ tiểu đường
        </div>

        <div class="hero-description">
            Hệ thống sử dụng mô hình Machine Learning
            XGBoost được xây dựng từ dữ liệu BRFSS 2023
            để hỗ trợ sàng lọc nguy cơ tiểu đường dựa trên
            các thông tin sức khỏe, nhân khẩu học
            và lối sống.
        </div>

    </div>
    """
)


st.warning(
    dedent(
        """
        ⚠️ **Lưu ý y tế:** Kết quả của hệ thống chỉ phục vụ
        mục đích sàng lọc và tham khảo.

        Đây không phải là chẩn đoán y khoa và không thay thế
        xét nghiệm hoặc tư vấn của nhân viên y tế.
        """
    )
)


# ============================================================
# TABS
# ============================================================

screening_tab, history_tab = st.tabs(
    [
        "🔎 Sàng lọc nguy cơ",
        "📋 Lịch sử sàng lọc"
    ]
)


# ============================================================
# TAB 1
# SCREENING
# ============================================================

with screening_tab:

    st.write(
        dedent(
            """
            Vui lòng nhập các thông tin bên dưới.
            Các thông tin này tương ứng với các đặc trưng
            được sử dụng bởi mô hình Machine Learning.
            """
        )
    )


    with st.form(
        "diabetes_screening_form"
    ):

        # ====================================================
        # 1. THÔNG TIN CƠ BẢN
        # ====================================================

        render_html(
            """
            <div class="section-box">

                <div class="section-title">
                    👤 1. Thông tin cơ bản
                </div>

                <div class="section-description">
                    Tuổi, giới tính, thể trạng
                    và thông tin nhân khẩu học.
                </div>

            </div>
            """
        )


        col1, col2 = st.columns(
            2
        )


        with col1:

            age = st.number_input(
                "Tuổi",
                min_value=18,
                max_value=110,
                value=40,
                step=1
            )


            sex = st.selectbox(
                "Giới tính",

                options=SEX_OPTIONS,

                index=get_default_index(
                    SEX_OPTIONS,
                    "Male"
                ),

                format_func=lambda x:
                    SEX_LABELS.get(
                        str(x),
                        str(x)
                    )
            )


            race = st.selectbox(
                "Nhóm chủng tộc / sắc tộc "
                "(theo phân nhóm BRFSS)",

                options=RACE_OPTIONS,

                format_func=lambda x:
                    RACE_LABELS.get(
                        str(x),
                        str(x)
                    )
            )


        with col2:

            height_cm = st.number_input(
                "Chiều cao (cm)",
                min_value=100.0,
                max_value=250.0,
                value=165.0,
                step=0.5
            )


            weight_kg = st.number_input(
                "Cân nặng (kg)",
                min_value=20.0,
                max_value=300.0,
                value=60.0,
                step=0.5
            )


            bmi = (
                weight_kg
                /
                (
                    height_cm
                    / 100
                ) ** 2
            )


            st.info(
                f"**BMI được tính tự động:** "
                f"{bmi:.2f}"
            )


        # ====================================================
        # 2. TÌNH TRẠNG SỨC KHỎE
        # ====================================================

        render_html(
            """
            <div class="section-box">

                <div class="section-title">
                    ❤️ 2. Tình trạng sức khỏe
                </div>

                <div class="section-description">
                    Các thông tin sức khỏe hiện tại
                    hoặc đã từng được bác sĩ
                    hay nhân viên y tế thông báo.
                </div>

            </div>
            """
        )


        health_col1, health_col2 = (
            st.columns(
                2
            )
        )


        with health_col1:

            general_health = (
                st.selectbox(
                    "Bạn tự đánh giá sức khỏe "
                    "tổng quát của mình như thế nào?",

                    options=list(
                        GENERAL_HEALTH_LABELS.keys()
                    ),

                    index=2,

                    format_func=lambda x:
                        GENERAL_HEALTH_LABELS[x]
                )
            )


            high_bp = st.selectbox(
                "Bạn có từng được bác sĩ hoặc nhân viên y tế "
                "thông báo rằng bạn bị cao huyết áp không?",

                options=BP_OPTIONS,

                index=get_default_index(
                    BP_OPTIONS,
                    "No"
                ),

                format_func=lambda x:
                    BP_LABELS.get(
                        str(x),
                        str(x)
                    )
            )


        with health_col2:

            high_cholesterol = (
                st.selectbox(
                    "Bạn có từng được bác sĩ hoặc nhân viên y tế "
                    "thông báo rằng cholesterol của bạn cao không?",

                    options=[
                        "No",
                        "Yes",
                        "Unknown"
                    ],

                    index=0,

                    format_func=lambda x:
                        YES_NO_UNKNOWN_LABELS[x]
                )
            )


            had_heart_attack = (
                st.selectbox(
                    "Bạn có từng được bác sĩ thông báo rằng "
                    "bạn đã bị nhồi máu cơ tim không?",

                    options=[
                        "No",
                        "Yes",
                        "Unknown"
                    ],

                    index=0,

                    format_func=lambda x:
                        YES_NO_UNKNOWN_LABELS[x]
                )
            )


        # ====================================================
        # 3. LỐI SỐNG & CHĂM SÓC Y TẾ
        # ====================================================

        render_html(
            """
            <div class="section-box">

                <div class="section-title">
                    🏥 3. Lối sống & chăm sóc y tế
                </div>

                <div class="section-description">
                    Thói quen, tình trạng việc làm
                    và mức độ tiếp cận dịch vụ chăm sóc sức khỏe.
                </div>

            </div>
            """
        )


        care_col1, care_col2 = (
            st.columns(
                2
            )
        )


        with care_col1:

            drinks_alcohol = (
                st.selectbox(
                    "Bạn có sử dụng đồ uống có cồn không?",

                    options=[
                        "No",
                        "Yes",
                        "Unknown"
                    ],

                    index=0,

                    format_func=lambda x:
                        YES_NO_UNKNOWN_LABELS[x]
                )
            )


            employment_status = (
                st.selectbox(
                    "Tình trạng việc làm",

                    options=EMPLOYMENT_OPTIONS,

                    format_func=lambda x:
                        EMPLOYMENT_LABELS.get(
                            str(x),
                            str(x)
                        )
                )
            )


        with care_col2:

            last_checkup = (
                st.selectbox(
                    "Lần khám sức khỏe định kỳ "
                    "gần nhất của bạn là khi nào?",

                    options=list(
                        CHECKUP_LABELS.keys()
                    ),

                    index=0,

                    format_func=lambda x:
                        CHECKUP_LABELS[x]
                )
            )


            has_personal_doctor = (
                st.selectbox(
                    "Bạn có bác sĩ hoặc nơi chăm sóc "
                    "sức khỏe thường xuyên không?",

                    options=DOCTOR_OPTIONS,

                    index=get_default_index(
                        DOCTOR_OPTIONS,
                        "Yes, only one"
                    ),

                    format_func=lambda x:
                        DOCTOR_LABELS.get(
                            str(x),
                            str(x)
                        )
                )
            )


        st.write("")


        submitted = (
            st.form_submit_button(
                "🔎 Thực hiện sàng lọc",
                width="stretch"
            )
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if submitted:

        input_data = {

            "age":
                int(age),

            "bmi":
                float(bmi),

            "general_health":
                general_health,

            "high_bp":
                high_bp,

            "high_cholesterol":
                high_cholesterol,

            "race":
                race,

            "last_checkup":
                last_checkup,

            "drinks_alcohol":
                drinks_alcohol,

            "sex":
                sex,

            "has_personal_doctor":
                has_personal_doctor,

            "employment_status":
                employment_status,

            "had_heart_attack":
                had_heart_attack
        }


        try:

            result = predict_and_save(
                input_data
            )


            st.session_state[
                "last_result"
            ] = result


            st.session_state[
                "last_input"
            ] = input_data


        except Exception as error:

            st.error(
                "Không thể thực hiện dự đoán "
                "hoặc lưu kết quả vào SQL Server."
            )


            st.exception(
                error
            )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    if (
        "last_result"
        in st.session_state
    ):

        result = (
            st.session_state[
                "last_result"
            ]
        )


        last_input = (
            st.session_state.get(
                "last_input",
                {}
            )
        )


        # ====================================================
        # SCORE CALCULATION
        # ====================================================

        score = float(
            result[
                "prediction_score"
            ]
        )


        threshold = float(
            result[
                "threshold"
            ]
        )


        score_percent = (
            score * 100
        )


        threshold_percent = (
            threshold * 100
        )


        difference_percent = (
            score_percent
            - threshold_percent
        )


        # Giới hạn thanh hiển thị trong 0 - 100
        bar_percent = max(
            0.0,
            min(
                score_percent,
                100.0
            )
        )


        st.divider()


        render_html(
            """
            <div class="result-title">
                📊 Kết quả sàng lọc
            </div>
            """
        )


        # ====================================================
        # METRICS
        # ====================================================

        metric1, metric2, metric3 = (
            st.columns(
                3
            )
        )


        metric1.metric(
            "Model Risk Score",
            f"{score_percent:.1f}%"
        )


        metric2.metric(
            "Ngưỡng sàng lọc",
            f"{threshold_percent:.1f}%"
        )


        metric3.metric(
            "Mã lần sàng lọc",
            result.get(
                "case_id",
                "-"
            )
        )


        # ====================================================
        # RISK SCORE CARD
        # ====================================================

        if score >= threshold:

            score_class = (
                "risk-score-alert"
            )


            bar_class = (
                "risk-bar-fill-alert"
            )


            status_text = (
                "Vượt ngưỡng sàng lọc"
            )


            comparison_text = (
                f"Cao hơn ngưỡng "
                f"{abs(difference_percent):.1f} "
                f"điểm phần trăm"
            )


        else:

            score_class = (
                "risk-score-safe"
            )


            bar_class = (
                "risk-bar-fill-safe"
            )


            status_text = (
                "Không vượt ngưỡng sàng lọc"
            )


            comparison_text = (
                f"Thấp hơn ngưỡng "
                f"{abs(difference_percent):.1f} "
                f"điểm phần trăm"
            )


        render_html(
            f"""
            <div class="risk-score-card {score_class}">

                <div class="risk-score-label">
                    Model Risk Score quy đổi
                </div>

                <div class="risk-score-value">
                    {score_percent:.1f}%
                </div>

                <div class="risk-status">
                    {status_text}
                </div>

                <div class="risk-bar">

                    <div
                        class="{bar_class}"
                        style="width: {bar_percent:.1f}%;">
                    </div>

                </div>

                <div class="risk-score-detail">

                    <b>Ngưỡng của mô hình:</b>
                    {threshold_percent:.1f}%

                    <br>

                    <b>So với ngưỡng:</b>
                    {comparison_text}

                </div>

                <div class="risk-score-note">

                    Phần trăm này là cách biểu diễn
                    Model Risk Score trên thang 0–100%.

                    Đây không phải là xác suất lâm sàng
                    mắc tiểu đường.

                </div>

            </div>
            """
        )


        # ====================================================
        # RESULT STATUS
        # ====================================================

        if (
            result[
                "predicted_class"
            ]
            == 1
        ):

            st.warning(
                dedent(
                    """
                    ⚠️ **Kết quả: Vượt ngưỡng sàng lọc**

                    Điểm do mô hình tạo ra cao hơn
                    ngưỡng sàng lọc đã được xác định
                    trên tập Validation.
                    """
                )
            )


        else:

            st.success(
                dedent(
                    """
                    ✅ **Kết quả: Không vượt ngưỡng sàng lọc**

                    Điểm do mô hình tạo ra chưa vượt
                    ngưỡng sàng lọc đã xác định.
                    """
                )
            )


        # ====================================================
        # RECOMMENDATION
        # ====================================================

        st.info(
            result[
                "recommendation"
            ]
        )


        # ====================================================
        # MISSING INFORMATION
        # ====================================================

        missing_information = []


        if (
            last_input.get(
                "high_cholesterol"
            )
            == "Unknown"
        ):

            missing_information.append(
                "thông tin cholesterol"
            )


        if (
            last_input.get(
                "drinks_alcohol"
            )
            == "Unknown"
        ):

            missing_information.append(
                "thông tin sử dụng đồ uống có cồn"
            )


        if (
            last_input.get(
                "had_heart_attack"
            )
            == "Unknown"
        ):

            missing_information.append(
                "tiền sử nhồi máu cơ tim"
            )


        if missing_information:

            st.warning(
                "Một số thông tin chưa được cung cấp đầy đủ: "
                + ", ".join(
                    missing_information
                )
                + ". "
                + "Mô hình vẫn có thể xử lý giá trị thiếu, "
                + "nhưng kết quả nên được diễn giải "
                + "thận trọng hơn."
            )


        # ====================================================
        # SCORE EXPLANATION
        # ====================================================

        render_html(
            """
            <div class="medical-note">

                <b>Giải thích Model Risk Score:</b>

                <br><br>

                Model Risk Score là điểm do mô hình
                Machine Learning tạo ra trên thang từ 0 đến 1.

                Trên giao diện, điểm này được nhân với 100
                để biểu diễn dưới dạng phần trăm cho dễ đọc.

                <br><br>

                Ví dụ:
                Score 0.65 được hiển thị thành 65%.

                Việc quy đổi này không làm thay đổi
                kết quả của mô hình.

                <br><br>

                Điểm được so sánh với ngưỡng sàng lọc
                của mô hình để xác định kết quả
                vượt hoặc không vượt ngưỡng.

                <br><br>

                <b>Quan trọng:</b>

                Phần trăm Model Risk Score không phải
                xác suất lâm sàng mắc tiểu đường.

                <br><br>

                Kết quả sàng lọc không phải chẩn đoán.
                Nếu có triệu chứng, yếu tố nguy cơ hoặc lo ngại
                về sức khỏe, người dùng nên trao đổi với
                nhân viên y tế hoặc thực hiện xét nghiệm phù hợp.

            </div>
            """
        )


# ============================================================
# TAB 2
# HISTORY
# ============================================================

with history_tab:

    st.subheader(
        "📋 Lịch sử sàng lọc"
    )


    st.write(
        dedent(
            """
            Các kết quả dưới đây được đọc trực tiếp
            từ SQL Server `DiabetesRiskDB`.
            """
        )
    )


    try:

        history_df = (
            load_prediction_history(
                limit=50
            )
        )


        if history_df.empty:

            st.info(
                "Chưa có dữ liệu sàng lọc."
            )


        else:

            display_df = (
                history_df.copy()
            )


            # =================================================
            # RENAME COLUMNS
            # =================================================

            display_df.rename(
                columns={

                    "prediction_id":
                        "Prediction ID",

                    "case_id":
                        "Case ID",

                    "age":
                        "Tuổi",

                    "bmi":
                        "BMI",

                    "sex":
                        "Giới tính",

                    "general_health":
                        "Sức khỏe chung",

                    "prediction_score":
                        "Model Risk Score (%)",

                    "threshold":
                        "Ngưỡng (%)",

                    "predicted_class":
                        "Class",

                    "risk_level":
                        "Kết quả",

                    "model_version":
                        "Phiên bản model",

                    "created_at":
                        "Thời gian"
                },

                inplace=True
            )


            # =================================================
            # VIỆT HÓA GIỚI TÍNH
            # =================================================

            if (
                "Giới tính"
                in display_df.columns
            ):

                display_df[
                    "Giới tính"
                ] = (

                    display_df[
                        "Giới tính"
                    ]

                    .map(
                        lambda x:
                            SEX_LABELS.get(
                                str(x),
                                str(x)
                            )
                    )
                )


            # =================================================
            # VIỆT HÓA SỨC KHỎE CHUNG
            # =================================================

            if (
                "Sức khỏe chung"
                in display_df.columns
            ):

                display_df[
                    "Sức khỏe chung"
                ] = (

                    display_df[
                        "Sức khỏe chung"
                    ]

                    .map(
                        lambda x:
                            GENERAL_HEALTH_LABELS.get(
                                str(x),
                                str(x)
                            )
                    )
                )


            # =================================================
            # SCORE -> %
            # =================================================

            if (
                "Model Risk Score (%)"
                in display_df.columns
            ):

                display_df[
                    "Model Risk Score (%)"
                ] = (

                    display_df[
                        "Model Risk Score (%)"
                    ]

                    * 100
                ).round(
                    1
                )


            # =================================================
            # THRESHOLD -> %
            # =================================================

            if (
                "Ngưỡng (%)"
                in display_df.columns
            ):

                display_df[
                    "Ngưỡng (%)"
                ] = (

                    display_df[
                        "Ngưỡng (%)"
                    ]

                    * 100
                ).round(
                    1
                )


            # =================================================
            # FORMAT BMI
            # =================================================

            if (
                "BMI"
                in display_df.columns
            ):

                display_df[
                    "BMI"
                ] = (

                    display_df[
                        "BMI"
                    ]

                    .round(
                        2
                    )
                )


            # =================================================
            # DISPLAY TABLE
            # =================================================

            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True
            )


            st.caption(
                f"Hiển thị {len(display_df)} "
                "lần sàng lọc gần nhất."
            )


    except Exception as error:

        st.error(
            "Không thể đọc lịch sử "
            "từ SQL Server."
        )


        st.exception(
            error
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Đồ án phân tích dữ liệu và Machine Learning "
    "hỗ trợ sàng lọc nguy cơ tiểu đường — BRFSS 2023."
)