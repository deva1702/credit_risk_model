import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from src.utils.logger import get_logger
from src.utils.config import MODEL_PATH, TARGET_COL, ID_COL

logger = get_logger(__name__)

# Columns to drop — too many missing or not useful
COLS_TO_DROP = [
    'SK_ID_CURR',
    'COMMONAREA_AVG', 'COMMONAREA_MODE', 'COMMONAREA_MEDI',
    'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAPARTMENTS_MEDI',
    'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'FLOORSMIN_AVG',
    'FLOORSMIN_MODE', 'FLOORSMIN_MEDI', 'YEARS_BUILD_AVG',
    'YEARS_BUILD_MODE', 'YEARS_BUILD_MEDI', 'LANDAREA_AVG',
    'LANDAREA_MODE', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_AVG',
    'LIVINGAPARTMENTS_MODE', 'LIVINGAPARTMENTS_MEDI',
    'NONLIVINGAREA_AVG', 'NONLIVINGAREA_MODE', 'NONLIVINGAREA_MEDI',
]


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    """Drop columns with more than threshold fraction of missing values."""
    missing_rate = df.isnull().mean()
    high_missing = missing_rate[missing_rate > threshold].index.tolist()
    high_missing = [c for c in high_missing if c != TARGET_COL]
    logger.info(f"Dropping {len(high_missing)} columns with >{threshold*100}% missing")
    return df.drop(columns=high_missing, errors='ignore')


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create meaningful derived features for credit risk."""
    df = df.copy()

    # Credit burden — how much credit relative to income
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)

    # Annuity burden — monthly payment relative to income
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)

    # Credit vs goods price — how much extra credit taken
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)

    # Age in years (DAYS_BIRTH is negative)
    df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).astype(int)

    # Employment length in years
    df['EMPLOYMENT_YEARS'] = (-df['DAYS_EMPLOYED'].clip(upper=0) / 365)

    # Employment to age ratio — stability indicator
    df['EMPLOYMENT_AGE_RATIO'] = df['EMPLOYMENT_YEARS'] / (df['AGE_YEARS'] + 1)

    # EXT_SOURCE mean — average of all 3 external scores
    ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
                if c in df.columns]
    if ext_cols:
        df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
        df['EXT_SOURCE_MIN']  = df[ext_cols].min(axis=1)
        df['EXT_SOURCE_PROD'] = df[ext_cols].prod(axis=1)

    # Income per family member
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)

    # Days since last phone change relative to employment
    if 'DAYS_LAST_PHONE_CHANGE' in df.columns:
        df['PHONE_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_EMPLOYED'] - 1)

    # Credit stress score — combines 3 signals
    df['CREDIT_STRESS'] = (
        df['CREDIT_INCOME_RATIO'] * 0.4 +
        df['ANNUITY_INCOME_RATIO'] * 0.4 +
        (1 - df.get('EXT_SOURCE_MEAN', pd.Series([0.5]*len(df)))) * 0.2
    )

    # Debt service ratio — monthly payment vs monthly income
    df['DEBT_SERVICE_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] / 12 + 1)

    # EXT_SOURCE weighted combo — higher weight on SOURCE_2 (most complete)
    if 'EXT_SOURCE_2' in df.columns:
        w1 = df.get('EXT_SOURCE_1', pd.Series([0.5]*len(df)))
        w2 = df['EXT_SOURCE_2']
        w3 = df.get('EXT_SOURCE_3', pd.Series([0.5]*len(df)))
        df['EXT_SOURCE_WEIGHTED'] = (w1 * 0.25 + w2 * 0.50 + w3 * 0.25)

    # Age-employment interaction — older + longer employed = more stable
    df['AGE_EMPLOYED_INTERACTION'] = df['AGE_YEARS'] * df['EMPLOYMENT_YEARS']

    logger.info(f"Engineered features — shape now: {df.shape}")
    return df


def encode_categoricals(df: pd.DataFrame,
                         encoders: dict = None,
                         fit: bool = True) -> tuple:
    """Label encode all categorical columns."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Encoding {len(cat_cols)} categorical columns")

    if fit:
        encoders = {}

    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = df[col].astype(str).fillna('Missing')
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).fillna('Missing')
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else 'Missing')
                df[col] = le.transform(df[col])

    return df, encoders


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining missing values."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


def preprocess(df: pd.DataFrame,
               fit: bool = True,
               encoders: dict = None) -> tuple:
    """
    Full preprocessing pipeline.
    fit=True  → training mode, fits and saves encoders
    fit=False → inference mode, uses saved encoders
    """
    logger.info(f"Starting preprocessing — fit={fit}")

    # Step 1: Drop high missing columns
    df = drop_high_missing(df)

    # Step 2: Drop explicitly listed columns
    df = df.drop(columns=COLS_TO_DROP, errors='ignore')

    # Step 3: Feature engineering
    df = engineer_features(df)

    # Step 4: Encode categoricals
    df, encoders = encode_categoricals(df, encoders=encoders, fit=fit)

    # Step 5: Fill remaining missing
    df = fill_missing(df)

    logger.info(f"Preprocessing complete — final shape: {df.shape}")
    return df, encoders


def get_train_test_split(df: pd.DataFrame) -> tuple:
    """Split into X_train, X_val, y_train, y_val."""
    X = df.drop(columns=[TARGET_COL], errors='ignore')
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
    logger.info(f"Train default rate: {y_train.mean()*100:.2f}%")
    logger.info(f"Val default rate:   {y_val.mean()*100:.2f}%")
    return X_train, X_val, y_train, y_val


def save_encoders(encoders: dict, path: str = None):
    path = path or os.path.join(MODEL_PATH, "encoders.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(encoders, path)
    logger.info(f"Encoders saved → {path}")


def load_encoders(path: str = None) -> dict:
    path = path or os.path.join(MODEL_PATH, "encoders.pkl")
    return joblib.load(path)