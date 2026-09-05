import json
import joblib
import xgboost as xgb


# ============================================================
# FILE PATHS
# ============================================================

PREPROCESSOR_PATH = (
    "models/brfss_preprocessor.joblib"
)

MODEL_PATH = (
    "models/brfss_xgboost_model.json"
)

METADATA_PATH = (
    "models/diabetes_model_metadata.json"
)


# ============================================================
# 1. LOAD METADATA
# ============================================================

print(
    "===================================="
)

print(
    "LOADING METADATA"
)

print(
    "===================================="
)


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


print(
    "Model:",
    metadata["model_name"]
)

print(
    "Feature set:",
    metadata["feature_set"]
)

print(
    "Threshold:",
    metadata["threshold"]
)

print(
    "\nFeatures:"
)


for i, feature in enumerate(
    metadata["features"],
    start=1
):

    print(
        f"{i:2}. {feature}"
    )


# ============================================================
# 2. LOAD PREPROCESSOR
# ============================================================

print(
    "\n===================================="
)

print(
    "LOADING PREPROCESSOR"
)

print(
    "===================================="
)


preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


print(
    "Load preprocessor thành công."
)

print(
    "Type:",
    type(preprocessor)
)


# ============================================================
# 3. LOAD XGBOOST
# ============================================================

print(
    "\n===================================="
)

print(
    "LOADING XGBOOST"
)

print(
    "===================================="
)


model = xgb.XGBClassifier()

model.load_model(
    MODEL_PATH
)


print(
    "Load XGBoost thành công."
)

print(
    "Type:",
    type(model)
)


# ============================================================
# 4. FINAL CHECK
# ============================================================

print(
    "\n===================================="
)

print(
    "DEPLOY MODEL READY"
)

print(
    "===================================="
)

print(
    "Preprocessor: OK"
)

print(
    "XGBoost: OK"
)

print(
    "Metadata: OK"
)

print(
    "Số feature:",
    len(metadata["features"])
)