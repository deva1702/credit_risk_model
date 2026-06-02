import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.data.loader import load_and_merge
from src.data.preprocessor import preprocess, get_train_test_split, load_encoders
from src.utils.logger import get_logger
from src.utils.config import MODEL_PATH

logger = get_logger("shap")
os.makedirs("notebooks/charts", exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)


def build_and_save_explainer():
    """Build SHAP TreeExplainer and save it. Run once after training."""
    logger.info("Loading model and data for SHAP...")

    model        = joblib.load(os.path.join(MODEL_PATH, "lgbm_model.pkl"))
    encoders     = load_encoders()
    feature_names = joblib.load(os.path.join(MODEL_PATH, "feature_names.pkl"))

    # Load a sample of validation data for background
    df = load_and_merge()
    df_processed, _ = preprocess(df, fit=False, encoders=encoders)
    _, X_val, _, y_val = get_train_test_split(df_processed)
    X_val = X_val[feature_names]

    # Use 500 samples — enough for stable SHAP values, fast to compute
    sample = X_val.sample(500, random_state=42)

    logger.info("Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    logger.info("Computing SHAP values on sample (this takes ~1-2 min)...")
    shap_values = explainer.shap_values(sample)

    # For binary classification lgbm returns list [class0, class1]
    # We want class 1 (default probability)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # ── Save explainer ─────────────────────────────────────
    explainer_path = os.path.join(MODEL_PATH, "shap_explainer.pkl")
    joblib.dump(explainer, explainer_path)
    logger.info(f"SHAP explainer saved → {explainer_path}")

    # ── Save sample + shap values for summary plot ─────────
    joblib.dump({
        "sample":      sample,
        "shap_values": shap_vals,
        "feature_names": feature_names,
    }, os.path.join(MODEL_PATH, "shap_data.pkl"))
    logger.info("SHAP data saved → models/shap_data.pkl")

    # ── Plot 1: Global feature importance (beeswarm) ───────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals, sample,
        feature_names=feature_names,
        max_display=20,
        show=False
    )
    plt.title("SHAP Summary — Top 20 Features driving Default Risk", pad=12)
    plt.tight_layout()
    plt.savefig("notebooks/charts/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved shap_summary.png")

    # ── Plot 2: Bar plot of mean |SHAP| ────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_vals, sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False
    )
    plt.title("Mean |SHAP Value| — Global Feature Importance", pad=12)
    plt.tight_layout()
    plt.savefig("notebooks/charts/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved shap_bar.png")

    logger.info("SHAP build complete")
    return explainer, shap_vals, sample, feature_names


def explain_single(input_df: pd.DataFrame,
                   explainer=None, feature_names=None) -> dict:
    """
    Explain a single prediction — returns top 5 risk factors.
    input_df: already preprocessed single-row DataFrame
    """
    if explainer is None:
        explainer = joblib.load(os.path.join(MODEL_PATH, "shap_explainer.pkl"))
    if feature_names is None:
        feature_names = joblib.load(os.path.join(MODEL_PATH, "feature_names.pkl"))

    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    # Build explanation dataframe
    explanation = pd.DataFrame({
        "feature":    feature_names,
        "shap_value": sv,
        "abs_value":  np.abs(sv),
    }).sort_values("abs_value", ascending=False).head(10)

    # Tag direction
    explanation["direction"] = explanation["shap_value"].apply(
        lambda x: "↑ Increases risk" if x > 0 else "↓ Decreases risk"
    )
    explanation["impact"] = explanation["shap_value"].apply(
        lambda x: "high" if abs(x) > 0.1 else "medium" if abs(x) > 0.05 else "low"
    )

    return explanation.to_dict(orient="records")


def plot_waterfall_single(input_df: pd.DataFrame,
                          explainer=None,
                          feature_names=None,
                          save_path: str = "notebooks/charts/waterfall.png"):
    """Generate waterfall plot for a single prediction."""
    if explainer is None:
        explainer = joblib.load(os.path.join(MODEL_PATH, "shap_explainer.pkl"))
    if feature_names is None:
        feature_names = joblib.load(os.path.join(MODEL_PATH, "feature_names.pkl"))

    shap_values = explainer(input_df)

    plt.figure(figsize=(10, 6))
    # Use index 1 for default class
    if shap_values.values.ndim == 3:
        sv = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1],
            data=shap_values.data,
            feature_names=feature_names
        )
    else:
        sv = shap_values

    shap.plots.waterfall(sv[0], max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


if __name__ == "__main__":
    build_and_save_explainer()