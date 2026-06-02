import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from src.data.loader import load_and_merge
from src.data.preprocessor import preprocess, get_train_test_split, save_encoders, load_encoders
from src.utils.logger import get_logger
from src.utils.config import MODEL_PATH

logger = get_logger("train")

LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'max_depth': 6,
    'num_leaves': 63,
    'min_child_samples': 100,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


def train():
    os.makedirs(MODEL_PATH, exist_ok=True)

    logger.info("Loading and merging dataset...")
    df = load_and_merge()

    logger.info("Preprocessing...")
    df_processed, encoders = preprocess(df, fit=True)

    X_train, X_val, y_train, y_val = get_train_test_split(df_processed)

    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, os.path.join(MODEL_PATH, "feature_names.pkl"))
    logger.info(f"Total features used: {len(feature_names)}")

    logger.info("Training LightGBM...")
    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=50),
        ]
    )

    best_iter = model.best_iteration_
    logger.info(f"Best iteration: {best_iter}")

    model_path = os.path.join(MODEL_PATH, "lgbm_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved → {model_path}")

    save_encoders(encoders)

    from sklearn.metrics import roc_auc_score
    val_preds = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)
    logger.info(f"Validation ROC-AUC: {val_auc:.4f}")

    return model, X_val, y_val, val_preds, feature_names


def train_xgboost():
    from xgboost import XGBClassifier
    os.makedirs(MODEL_PATH, exist_ok=True)

    logger.info("Loading and merging dataset for XGBoost...")
    df = load_and_merge()

    logger.info("Preprocessing with existing encoders...")
    df_processed, encoders = preprocess(df, fit=False,
                                        encoders=load_encoders())

    X_train, X_val, y_train, y_val = get_train_test_split(df_processed)
    feature_names = joblib.load(os.path.join(MODEL_PATH, "feature_names.pkl"))
    X_train = X_train[feature_names]
    X_val   = X_val[feature_names]

    logger.info("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc',
        early_stopping_rounds=100,
        verbosity=0,
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    from sklearn.metrics import roc_auc_score
    val_preds = xgb_model.predict_proba(X_val)[:, 1]
    val_auc   = roc_auc_score(y_val, val_preds)
    logger.info(f"XGBoost Validation ROC-AUC: {val_auc:.4f}")

    model_path = os.path.join(MODEL_PATH, "xgb_model.pkl")
    joblib.dump(xgb_model, model_path)
    logger.info(f"XGBoost model saved → {model_path}")

    return xgb_model, X_val, y_val, val_preds


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "xgb":
        train_xgboost()
    else:
        train()