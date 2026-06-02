import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
DATA_PATH     = os.getenv("DATA_PATH", "./data")
MODEL_PATH    = os.getenv("MODEL_PATH", "./models")
DB_PATH       = os.getenv("DB_PATH", "./sql/credit_risk.db")
ACTIVE_MODEL  = os.getenv("ACTIVE_MODEL", "lgbm")  # lgbm or xgb

GROQ_MODEL = "llama-3.3-70b-versatile"

RISK_THRESHOLDS = {
    "low":    0.3,
    "medium": 0.6,
}

TARGET_COL = "TARGET"
ID_COL     = "SK_ID_CURR"