import pandas as pd
import os
from src.utils.logger import get_logger
from src.utils.config import DATA_PATH

logger = get_logger(__name__)

def load_main_table(filename="application_train.csv") -> pd.DataFrame:
    path = os.path.join(DATA_PATH, filename)
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df

def load_bureau() -> pd.DataFrame:
    path = os.path.join(DATA_PATH, "bureau.csv")
    logger.info(f"Loading bureau data (Optimized 10/17 Columns)")
    use_cols = [
        "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT", 
        "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", 
        "AMT_CREDIT_SUM_OVERDUE", "DAYS_CREDIT_ENDDATE", "CNT_CREDIT_PROLONG"
    ]
    return pd.read_csv(path, usecols=use_cols)

def load_previous_application() -> pd.DataFrame:
    path = os.path.join(DATA_PATH, "previous_application.csv")
    logger.info(f"Loading previous_application data (Optimized 5/37 Columns)")
    use_cols = ["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS", "AMT_CREDIT", "AMT_ANNUITY"]
    return pd.read_csv(path, usecols=use_cols)

def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    agg = bureau.groupby("SK_ID_CURR").agg(
        bureau_loan_count=("SK_ID_BUREAU", "count"),
        bureau_active_loans=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        bureau_closed_loans=("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
        bureau_avg_days_credit=("DAYS_CREDIT", "mean"),
        bureau_max_overdue=("AMT_CREDIT_MAX_OVERDUE", "max"),
        bureau_total_credit=("AMT_CREDIT_SUM", "sum"),
        bureau_total_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_total_overdue=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bureau_avg_days_enddate=("DAYS_CREDIT_ENDDATE", "mean"),
        bureau_num_prolong=("CNT_CREDIT_PROLONG", "sum"),
    ).reset_index()

    # Derived ratios — these are the powerful features
    agg['bureau_debt_ratio'] = agg['bureau_total_debt'] / (agg['bureau_total_credit'] + 1)
    agg['bureau_overdue_ratio'] = agg['bureau_total_overdue'] / (agg['bureau_total_credit'] + 1)
    agg['bureau_active_ratio'] = agg['bureau_active_loans'] / (agg['bureau_loan_count'] + 1)
    return agg

def aggregate_previous_app(prev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous applications per applicant."""
    agg = prev.groupby("SK_ID_CURR").agg(
        prev_app_count=("SK_ID_PREV", "count"),
        prev_approved_count=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        prev_refused_count=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        prev_avg_credit=("AMT_CREDIT", "mean"),
        prev_avg_annuity=("AMT_ANNUITY", "mean"),
    ).reset_index()
    return agg

def load_and_merge() -> pd.DataFrame:
    """Load main table and join bureau + previous_app aggregates."""
    df = load_main_table()
    
    try:
        bureau = load_bureau()
        bureau_agg = aggregate_bureau(bureau)
        df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")
        logger.info(f"After bureau merge: {df.shape[1]} cols")
    except FileNotFoundError:
        logger.warning("bureau.csv not found — skipping bureau features")

    try:
        prev = load_previous_application()
        prev_agg = aggregate_previous_app(prev)
        df = df.merge(prev_agg, on="SK_ID_CURR", how="left")
        logger.info(f"After previous_app merge: {df.shape[1]} cols")
    except FileNotFoundError:
        logger.warning("previous_application.csv not found — skipping prev_app features")

    return df