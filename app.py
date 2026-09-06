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
    page_title="Sàng lọc nguy cơ tiểu đường",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# HELPER
# ============================================================

def render_html(content):
    st.html(
        dedent(content)
    )


# ============================================================
# STYLE
# ============================================================

render_html(
    """
    <style>

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

    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
    }

    div[data-testid="stFormSubmitButton"] button {
        height: 48px;

        border-radius: 10px;

        font-size: 1rem;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        padding: 1rem 1.2rem;

        background: white;

        border: 1px solid #e2e8f0;
        border-radius: 14px;

        box-shadow:
            0 3px 12px
            rgba(15, 23, 42, 0.04);
    }

    .result-title {
        margin-top: 1rem;
        margin-bottom: 0.8rem;

        font-size: 1.35rem;
        font-weight: 700;

        color: #0f172a;
    }

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

    </style>
    """
)


# ============================================================
# ĐỌC CATEGORY TỪ PREPROCESSOR
# ============================================================

def get_fitted_categories(column_name):

    for _, transformer, columns in preprocessor.transformers_:

        try:
            column_list = list(columns)

        except TypeError:
            continue


        if column_name not in column_list:
            continue


        encoder = None


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

    values = get_fitted_categories(
        column_name
    )


    if not values:

        st.error(
            "Không thể đọc dữ liệu cần thiết "
            "từ mô hình."
        )

        st.stop()


    return values


def get_default_index(
    options,
    preferred_value
):

    try:

        return list(options).index(
            preferred_value
        )

    except ValueError:

        return 0


# ============================================================
# NHÃN TIẾNG VIỆT
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
        "Mức ranh giới",

    "Unknown":
        "Không biết / chưa được thông báo"
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

    "Out of work for less that 1 year":
        "Thất nghiệp dưới 1 năm",

    "Out of work for more than 1 year":
        "Thất nghiệp trên 1 năm",

    "Out of work for less than 1 year":
        "Thất nghiệp dưới 1 năm",

    "Out of work for 1 year or more":
        "Thất nghiệp từ 1 năm trở lên",

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


BP_INPUT_OPTIONS = list(
    BP_OPTIONS
)


if "Unknown" not in BP_INPUT_OPTIONS:

    BP_INPUT_OPTIONS.append(
        "Unknown"
    )


# ============================================================
# LỊCH SỬ SQL
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
            Sàng lọc nguy cơ tiểu đường
        </div>

        <div class="hero-title">
            🩺 Hệ thống sàng lọc nguy cơ tiểu đường
        </div>

        <div class="hero-description">

            Hệ thống sử dụng Machine Learning
            để hỗ trợ đánh giá nguy cơ tiểu đường
            dựa trên một số thông tin sức khỏe
            và thông tin cá nhân cơ bản.

        </div>

    </div>
    """
)


st.warning(
    "⚠️ Kết quả chỉ có mục đích hỗ trợ sàng lọc, "
    "không thay thế chẩn đoán hoặc xét nghiệm y khoa."
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
# TAB 1 — SÀNG LỌC
# ============================================================

with screening_tab:

    st.write(
        "Vui lòng cung cấp các thông tin bên dưới "
        "để hệ thống thực hiện sàng lọc."
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
                    Tuổi, giới tính, chiều cao,
                    cân nặng và nhóm nhân khẩu học.
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
                "Nhóm chủng tộc / sắc tộc",

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
                f"BMI của bạn: **{bmi:.2f}**"
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
                    Một số thông tin về sức khỏe
                    và tiền sử bệnh.
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

                options=BP_INPUT_OPTIONS,

                index=get_default_index(
                    BP_INPUT_OPTIONS,
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
        # 3. CHĂM SÓC Y TẾ & VIỆC LÀM
        # ====================================================

        render_html(
            """
            <div class="section-box">

                <div class="section-title">
                    🏥 3. Chăm sóc y tế & việc làm
                </div>

                <div class="section-description">
                    Thông tin về việc làm
                    và khả năng tiếp cận chăm sóc sức khỏe.
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
                "🔎 Xem kết quả sàng lọc",
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
                "Không thể thực hiện sàng lọc. "
                "Vui lòng thử lại."
            )


            st.exception(
                error
            )


    # ========================================================
    # HIỂN THỊ KẾT QUẢ
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
            score
            * 100
        )


        threshold_percent = (
            threshold
            * 100
        )


        display_score_percent = round(
            score_percent,
            1
        )


        display_threshold_percent = round(
            threshold_percent,
            1
        )


        difference_percent = (
            display_score_percent
            - display_threshold_percent
        )


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


        metric1, metric2, metric3 = (
            st.columns(
                3
            )
        )


        metric1.metric(
            "Điểm sàng lọc nguy cơ",
            f"{display_score_percent:.1f}%"
        )


        metric2.metric(
            "Ngưỡng sàng lọc",
            f"{display_threshold_percent:.1f}%"
        )


        metric3.metric(
            "Mã lần sàng lọc",
            result.get(
                "case_id",
                "-"
            )
        )


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
                    Điểm sàng lọc nguy cơ
                </div>

                <div class="risk-score-value">
                    {display_score_percent:.1f}%
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

                    <b>Ngưỡng sàng lọc:</b>
                    {display_threshold_percent:.1f}%

                    <br>

                    <b>So với ngưỡng:</b>
                    {comparison_text}

                </div>

                <div class="risk-score-note">
                    Điểm này dùng để hỗ trợ sàng lọc
                    và không phải xác suất chính xác
                    bạn mắc tiểu đường.
                </div>

            </div>
            """
        )


        if (
            result[
                "predicted_class"
            ]
            == 1
        ):

            st.warning(
                "⚠️ **Kết quả: Vượt ngưỡng sàng lọc**\n\n"
                "Bạn nên cân nhắc thực hiện xét nghiệm "
                "đường huyết hoặc HbA1c và trao đổi "
                "với nhân viên y tế."
            )


        else:

            st.success(
                "✅ **Kết quả: Không vượt ngưỡng sàng lọc**\n\n"
                "Kết quả hiện tại chưa vượt ngưỡng "
                "cảnh báo của hệ thống."
            )


        # ====================================================
        # THÔNG TIN CÒN THIẾU
        # ====================================================

        missing_information = []


        if (
            last_input.get(
                "high_bp"
            )
            == "Unknown"
        ):

            missing_information.append(
                "cao huyết áp"
            )


        if (
            last_input.get(
                "high_cholesterol"
            )
            == "Unknown"
        ):

            missing_information.append(
                "cholesterol"
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

            st.info(
                "Một số thông tin chưa được cung cấp: "
                + ", ".join(
                    missing_information
                )
                + "."
            )


# ============================================================
# TAB 2 — LỊCH SỬ
# ============================================================

with history_tab:

    st.subheader(
        "📋 Lịch sử sàng lọc"
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


            display_df.rename(
                columns={

                    "prediction_id":
                        "Mã kết quả",

                    "case_id":
                        "Mã sàng lọc",

                    "age":
                        "Tuổi",

                    "bmi":
                        "BMI",

                    "sex":
                        "Giới tính",

                    "general_health":
                        "Sức khỏe chung",

                    "prediction_score":
                        "Điểm sàng lọc nguy cơ (%)",

                    "threshold":
                        "Ngưỡng (%)",

                    "predicted_class":
                        "Phân loại",

                    "risk_level":
                        "Kết quả",

                    "model_version":
                        "Phiên bản mô hình",

                    "created_at":
                        "Thời gian"
                },

                inplace=True
            )


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


            if (
                "Điểm sàng lọc nguy cơ (%)"
                in display_df.columns
            ):

                display_df[
                    "Điểm sàng lọc nguy cơ (%)"
                ] = (

                    display_df[
                        "Điểm sàng lọc nguy cơ (%)"
                    ]

                    * 100
                ).round(
                    1
                )


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
            "Không thể đọc lịch sử sàng lọc."
        )


        st.exception(
            error
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Hệ thống hỗ trợ sàng lọc nguy cơ tiểu đường "
    "dựa trên mô hình Machine Learning."
)