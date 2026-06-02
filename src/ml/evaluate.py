import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
from src.data.loader import load_and_merge
from src.data.preprocessor import preprocess, get_train_test_split, load_encoders
from src.utils.logger import get_logger
from src.utils.config import MODEL_PATH, RISK_THRESHOLDS

logger = get_logger("evaluate")

os.makedirs("notebooks/charts", exist_ok=True)


def get_risk_band(prob: float) -> str:
    if prob < RISK_THRESHOLDS['low']:
        return 'Low'
    elif prob < RISK_THRESHOLDS['medium']:
        return 'Medium'
    return 'High'


def ks_statistic(y_true, y_prob) -> float:
    """KS statistic — banking industry standard metric."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='#1D9E75', lw=2, label=f'LightGBM (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random baseline')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#1D9E75')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Credit Default Prediction')
    ax.legend(loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('notebooks/charts/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved roc_curve.png")


def plot_pr_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color='#378ADD', lw=2,
            label=f'LightGBM (PR-AUC = {pr_auc:.4f})')
    ax.axhline(y=0.08, color='red', linestyle='--', lw=1, label='Random baseline (8%)')
    ax.fill_between(recall, precision, alpha=0.1, color='#378ADD')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — Credit Default Prediction')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('notebooks/charts/pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved pr_curve.png")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Greens')
    plt.colorbar(im)
    classes = ['No Default', 'Default']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix (threshold = 0.3)')
    plt.tight_layout()
    plt.savefig('notebooks/charts/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved confusion_matrix.png")


def plot_feature_importance(model, feature_names, top_n=20):
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(feat_imp['feature'], feat_imp['importance'],
                   color='#534AB7', edgecolor='white', linewidth=0.5)
    ax.set_title(f'Top {top_n} Feature Importances — LightGBM', fontsize=13)
    ax.set_xlabel('Importance score')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('notebooks/charts/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved feature_importance.png")


def evaluate():
    # ── Load model & encoders ──────────────────────────────
    model_path = os.path.join(MODEL_PATH, "lgbm_model.pkl")
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    encoders = load_encoders()
    feature_names = joblib.load(os.path.join(MODEL_PATH, "feature_names.pkl"))

    # ── Reload and reprocess data ──────────────────────────
    logger.info("Reloading data for evaluation...")
    df = load_and_merge()
    df_processed, _ = preprocess(df, fit=False, encoders=encoders)
    _, X_val, _, y_val = get_train_test_split(df_processed)

    # Align columns
    X_val = X_val[feature_names]

    # ── Get predictions ────────────────────────────────────
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred_30 = (y_prob >= 0.3).astype(int)  # 0.3 threshold — better recall

    # ── Compute all metrics ────────────────────────────────
    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    ks = ks_statistic(y_val, y_prob)

    # ── Risk band distribution ─────────────────────────────
    bands = pd.Series(y_prob).apply(get_risk_band).value_counts()

    # ── Print full report ──────────────────────────────────
    print("\n" + "="*55)
    print("       CREDIT RISK MODEL — EVALUATION REPORT")
    print("="*55)
    print(f"  ROC-AUC Score     : {roc_auc:.4f}")
    print(f"  PR-AUC Score      : {pr_auc:.4f}")
    print(f"  KS Statistic      : {ks:.4f}")
    print("-"*55)
    print("  Classification Report (threshold = 0.3):")
    print(classification_report(y_val, y_pred_30,
                                target_names=['No Default', 'Default']))
    print("-"*55)
    print("  Risk Band Distribution (validation set):")
    for band, count in bands.items():
        print(f"    {band:8s}: {count:6,} ({count/len(y_prob)*100:.1f}%)")
    print("="*55)

    # ── Save metrics as json ───────────────────────────────
    import json
    metrics = {
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'ks_statistic': round(ks, 4),
        'threshold_used': 0.3,
        'val_size': len(y_val),
    }
    with open(os.path.join(MODEL_PATH, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → models/metrics.json")

    # ── Generate all plots ─────────────────────────────────
    plot_roc_curve(y_val, y_prob)
    plot_pr_curve(y_val, y_prob)
    plot_confusion_matrix(y_val, y_pred_30)
    plot_feature_importance(model, feature_names)

    return metrics


def evaluate_xgboost():
    """Evaluate XGBoost model and compare with LightGBM."""
    import json

    # Load XGBoost model
    xgb_path = os.path.join(MODEL_PATH, "xgb_model.pkl")
    logger.info(f"Loading XGBoost model from {xgb_path}")
    model    = joblib.load(xgb_path)
    encoders = load_encoders()
    feature_names = joblib.load(os.path.join(MODEL_PATH, "feature_names.pkl"))

    # Reload data
    logger.info("Reloading data for XGBoost evaluation...")
    df = load_and_merge()
    df_processed, _ = preprocess(df, fit=False, encoders=encoders)
    _, X_val, _, y_val = get_train_test_split(df_processed)
    X_val = X_val[feature_names]

    # Predictions
    y_prob     = model.predict_proba(X_val)[:, 1]
    y_pred_30  = (y_prob >= 0.3).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc  = average_precision_score(y_val, y_prob)
    ks      = ks_statistic(y_val, y_prob)
    bands   = pd.Series(y_prob).apply(get_risk_band).value_counts()

    print("\n" + "="*55)
    print("       XGBOOST MODEL — EVALUATION REPORT")
    print("="*55)
    print(f"  ROC-AUC Score     : {roc_auc:.4f}")
    print(f"  PR-AUC Score      : {pr_auc:.4f}")
    print(f"  KS Statistic      : {ks:.4f}")
    print("-"*55)
    print("  Classification Report (threshold = 0.3):")
    print(classification_report(y_val, y_pred_30,
                                target_names=['No Default', 'Default']))
    print("-"*55)
    print("  Risk Band Distribution (validation set):")
    for band, count in bands.items():
        print(f"    {band:8s}: {count:6,} ({count/len(y_prob)*100:.1f}%)")
    print("="*55)

    # Save XGBoost metrics
    xgb_metrics = {
        'roc_auc':      round(roc_auc, 4),
        'pr_auc':       round(pr_auc, 4),
        'ks_statistic': round(ks, 4),
        'threshold_used': 0.3,
        'val_size':     len(y_val),
    }
    with open(os.path.join(MODEL_PATH, "xgb_metrics.json"), 'w') as f:
        json.dump(xgb_metrics, f, indent=2)
    logger.info("XGBoost metrics saved → models/xgb_metrics.json")

    # Print comparison
    lgbm_metrics_path = os.path.join(MODEL_PATH, "metrics.json")
    if os.path.exists(lgbm_metrics_path):
        with open(lgbm_metrics_path) as f:
            lgbm_metrics = json.load(f)
        print("\n" + "="*55)
        print("       MODEL COMPARISON")
        print("="*55)
        print(f"  {'Metric':<20} {'LightGBM':>10} {'XGBoost':>10} {'Winner':>10}")
        print("-"*55)
        for metric in ['roc_auc', 'pr_auc', 'ks_statistic']:
            lgbm_val = lgbm_metrics[metric]
            xgb_val  = xgb_metrics[metric]
            winner   = 'LightGBM' if lgbm_val >= xgb_val else 'XGBoost'
            print(f"  {metric:<20} {lgbm_val:>10.4f} {xgb_val:>10.4f} {winner:>10}")
        print("="*55)

    return xgb_metrics


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "xgb":
        evaluate_xgboost()
    else:
        evaluate()