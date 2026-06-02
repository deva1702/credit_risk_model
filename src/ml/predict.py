import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.utils.logger import get_logger
from src.utils.config import MODEL_PATH, RISK_THRESHOLDS

logger = get_logger("predict")


def load_model_artifacts(model_type: str = None):
    """Load model, encoders and feature names. model_type: lgbm or xgb"""
    from src.utils.config import ACTIVE_MODEL
    model_type = model_type or ACTIVE_MODEL

    model_file = f"{model_type}_model.pkl"
    model_path = os.path.join(MODEL_PATH, model_file)

    if not os.path.exists(model_path):
        logger.warning(f"{model_file} not found — falling back to lgbm")
        model_path = os.path.join(MODEL_PATH, "lgbm_model.pkl")

    model    = joblib.load(model_path)
    encoders = joblib.load(os.path.join(MODEL_PATH, "encoders.pkl"))
    features = joblib.load(os.path.join(MODEL_PATH, "feature_names.pkl"))
    logger.info(f"Loaded model: {model_type}")
    return model, encoders, features


def get_risk_band(prob: float) -> str:
    if prob < RISK_THRESHOLDS["low"]:
        return "Low"
    elif prob < RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def get_risk_color(band: str) -> str:
    return {"Low": "#1D9E75", "Medium": "#EF9F27", "High": "#E24B4A"}[band]


def preprocess_single(input_dict: dict, encoders: dict, feature_names: list) -> pd.DataFrame:
    """
    Preprocess a single applicant input dict into model-ready DataFrame.
    Handles missing features gracefully with median fill.
    """
    df = pd.DataFrame([input_dict])

    # ── Feature engineering (mirror preprocessor.py) ──────
    df['CREDIT_INCOME_RATIO']   = df['AMT_CREDIT']       / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO']  = df['AMT_ANNUITY']      / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_GOODS_RATIO']    = df['AMT_CREDIT']       / (df.get('AMT_GOODS_PRICE', pd.Series([1])) + 1)
    df['AGE_YEARS']             = (-df['DAYS_BIRTH']     / 365).astype(int)
    df['EMPLOYMENT_YEARS']      = (-df['DAYS_EMPLOYED'].clip(upper=0) / 365)
    df['EMPLOYMENT_AGE_RATIO']  = df['EMPLOYMENT_YEARS'] / (df['AGE_YEARS'] + 1)
    df['INCOME_PER_PERSON']     = df['AMT_INCOME_TOTAL'] / (df.get('CNT_FAM_MEMBERS', pd.Series([1])) + 1)

    ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
    if ext_cols:
        df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
        df['EXT_SOURCE_MIN']  = df[ext_cols].min(axis=1)
        df['EXT_SOURCE_PROD'] = df[ext_cols].prod(axis=1)

    # ── Missing engineered features added here ─────────────
    df['CREDIT_STRESS'] = (
        df['CREDIT_INCOME_RATIO'] * 0.4 +
        df['ANNUITY_INCOME_RATIO'] * 0.4 +
        (1 - df.get('EXT_SOURCE_MEAN', pd.Series([0.5]*len(df)))) * 0.2
    )
    
    df['DEBT_SERVICE_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] / 12 + 1)
    
    if 'EXT_SOURCE_2' in df.columns:
        w1 = df.get('EXT_SOURCE_1', pd.Series([0.5]*len(df)))
        w2 = df['EXT_SOURCE_2']
        w3 = df.get('EXT_SOURCE_3', pd.Series([0.5]*len(df)))
        df['EXT_SOURCE_WEIGHTED'] = (w1 * 0.25 + w2 * 0.50 + w3 * 0.25)
        
    df['AGE_EMPLOYED_INTERACTION'] = df['AGE_YEARS'] * df['EMPLOYMENT_YEARS']
    
    if 'DAYS_LAST_PHONE_CHANGE' in df.columns:
        df['PHONE_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_EMPLOYED'] - 1)
    else:
        df['PHONE_EMPLOY_RATIO'] = 0.0

    # ── Encode categoricals ────────────────────────────────
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        if col in encoders:
            le = encoders[col]
            df[col] = df[col].astype(str)
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    # ── Align to training feature set ─────────────────────
    # Add missing columns as 0, drop extra columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # ── Fill any remaining NaN ─────────────────────────────
    df = df.fillna(0)

    return df


def predict_single(input_dict: dict,
                   model=None, encoders=None, feature_names=None) -> dict:
    """
    Run prediction for a single applicant.
    Returns probability, risk band, and a clean result dict.
    """
    if model is None:
        model, encoders, feature_names = load_model_artifacts()

    X = preprocess_single(input_dict, encoders, feature_names)
    prob = float(model.predict_proba(X)[:, 1][0])
    band = get_risk_band(prob)
    color = get_risk_color(band)

    result = {
        "probability":    round(prob, 4),
        "risk_score":     round(prob * 100, 1),   # 0–100 scale for UI
        "risk_band":      band,
        "risk_color":     color,
        "approve":        band != "High",
        "confidence":     round(max(prob, 1 - prob) * 100, 1),
    }

    logger.info(f"Prediction → prob={prob:.4f} band={band}")
    return result


# ── Quick smoke test ───────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "AMT_INCOME_TOTAL":   180000,
        "AMT_CREDIT":         450000,
        "AMT_ANNUITY":        22500,
        "AMT_GOODS_PRICE":    450000,
        "DAYS_BIRTH":         -12000,   # ~33 years old
        "DAYS_EMPLOYED":      -2000,    # ~5.5 years employed
        "EXT_SOURCE_1":       0.6,
        "EXT_SOURCE_2":       0.7,
        "EXT_SOURCE_3":       0.5,
        "CODE_GENDER":        "M",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CNT_FAM_MEMBERS":    2,
        "NAME_INCOME_TYPE":   "Working",
        "NAME_EDUCATION_TYPE":"Secondary / secondary special",
    }

    result = predict_single(sample)
    print("\n── Prediction Result ──────────────────────────")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")