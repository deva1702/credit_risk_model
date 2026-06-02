import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Page config — must be first Streamlit call ─────────────
st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ────────────────────────────────────────────────
from src.ml.predict import load_model_artifacts, predict_single, preprocess_single
from src.ml.shap_explainer import explain_single, plot_waterfall_single
from src.talk_to_data.nl_to_sql import ask
from src.utils.helpers import build_sqlite_db
from src.utils.config import MODEL_PATH
from src.talk_to_data.prompt_templates import SAMPLE_QUERIES

# ── Global CSS ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #1a1d27; }
.metric-card {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-bottom: 8px;
}
.metric-val { font-size: 2rem; font-weight: 600; }
.metric-lbl { font-size: 0.8rem; color: #8b8fa8; margin-top: 4px; }
.risk-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 20px;
    font-size: 1.1rem;
    font-weight: 600;
}
.insight-box {
    background: #1e2130;
    border-left: 3px solid #4f8ef7;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #c8cad8;
}
.sql-box {
    background: #161b2e;
    border: 1px solid #2d3148;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: monospace;
    font-size: 0.8rem;
    color: #7dd3fc;
    margin: 6px 0;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #1e2130;
    border-radius: 8px;
    padding: 8px 20px;
    color: #8b8fa8;
}
.stTabs [aria-selected="true"] {
    background: #4f8ef7 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ── Cache heavy resources ──────────────────────────────────
@st.cache_resource
def load_artifacts(model_type: str = "lgbm"):
    model, encoders, features = load_model_artifacts(model_type)
    explainer = joblib.load(os.path.join(MODEL_PATH, "shap_explainer.pkl"))
    return model, encoders, features, explainer

@st.cache_data
def load_all_metrics():
    lgbm_path = os.path.join(MODEL_PATH, "metrics.json")
    xgb_path = os.path.join(MODEL_PATH, "xgb_metrics.json")
    
    with open(lgbm_path) as f:
        lgbm = json.load(f)
        
    xgb = None
    if os.path.exists(xgb_path):
        with open(xgb_path) as f:
            xgb = json.load(f)
            
    return lgbm, xgb

@st.cache_data
def load_shap_charts():
    return {
        "summary": "notebooks/charts/shap_summary.png",
        "bar":     "notebooks/charts/shap_bar.png",
        "roc":     "notebooks/charts/roc_curve.png",
        "pr":      "notebooks/charts/pr_curve.png",
        "cm":      "notebooks/charts/confusion_matrix.png",
        "fi":      "notebooks/charts/feature_importance.png",
    }

@st.cache_resource
def init_db():
    build_sqlite_db()
    return True


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Credit Risk Platform")
    st.markdown("*AI-Powered Loan Default Prediction*")
    st.divider()

    lgbm_metrics, xgb_metrics = load_all_metrics()
    st.markdown("### Model Performance")
    if xgb_metrics:
        st.markdown(f"""
| Metric | 🌿 LGBM | ⚡ XGB |
| :--- | :---: | :---: |
| **ROC-AUC** | `{lgbm_metrics['roc_auc']:.4f}` | `{xgb_metrics['roc_auc']:.4f}` |
| **KS Stat** | `{lgbm_metrics['ks_statistic']:.4f}` | `{xgb_metrics['ks_statistic']:.4f}` |
| **PR-AUC** | `{lgbm_metrics['pr_auc']:.4f}` | `{xgb_metrics['pr_auc']:.4f}` |
""")
    else:
        st.metric("ROC-AUC",      f"{lgbm_metrics['roc_auc']:.4f}")
        st.metric("KS Statistic", f"{lgbm_metrics['ks_statistic']:.4f}")
        st.metric("PR-AUC",       f"{lgbm_metrics['pr_auc']:.4f}")
    st.divider()

    st.markdown("### Dataset Info")
    st.markdown("- **307,511** loan applications")
    st.markdown("- **122** raw features")
    st.markdown("- **8.07%** default rate")
    st.markdown("- **3 tables** in SQLite DB")
    st.divider()
    st.markdown("*LightGBM + XGBoost · Groq LLaMA-3.3*")


# ── Main tabs ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA & Insights",
    "🎯 Risk Predictor",
    "🔍 Explainability",
    "💬 Talk to Data",
])


# ══════════════════════════════════════════════════════════
# TAB 1 — EDA & Insights
# ══════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Key patterns discovered in the Home Credit Default Risk dataset.")

    insights = [
        "🔵 Default Rate: 8.07% of applicants defaulted — severe class imbalance requiring scale_pos_weight correction",
        "🟢 Gender Signal: Male applicants default at ~10.1% vs ~7.0% for females — a statistically significant difference",
        "🟡 Age Risk: Applicants aged 20–30 have 2x the default rate of those aged 50–60 — youth correlates with risk",
        "🔴 Employment: Newly employed (0–2 yrs) and unemployed applicants show the highest default probability",
        "🟣 Income Band: Applicants earning below ₹100K annually default at nearly double the rate of higher earners",
        "⚪ EXT_SOURCE: External credit scores (EXT_SOURCE_1/2/3) are the strongest predictors — dominate SHAP values",
    ]
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    chart_files = [
        ("notebooks/charts/01_class_distribution.png",  "Class Distribution"),
        ("notebooks/charts/02_default_by_gender.png",   "Default Rate by Gender"),
        ("notebooks/charts/03_default_by_contract.png", "Default Rate by Contract Type"),
        ("notebooks/charts/04_default_by_income.png",   "Default Rate by Income Band"),
        ("notebooks/charts/05_default_by_age.png",      "Default Rate by Age Band"),
        ("notebooks/charts/06_default_by_employment.png","Default Rate by Employment"),
        ("notebooks/charts/07_missing_values.png",      "Missing Value Analysis"),
        ("notebooks/charts/08_correlation_heatmap.png", "Feature Correlation Heatmap"),
    ]
    for i, (path, title) in enumerate(chart_files):
        col = col1 if i % 2 == 0 else col2
        with col:
            if os.path.exists(path):
                st.markdown(f"**{title}**")
                st.image(path, use_container_width=True)
            else:
                st.warning(f"Chart not found: {title}")


# ══════════════════════════════════════════════════════════
# TAB 2 — Risk Predictor
# ══════════════════════════════════════════════════════════
with tab2:
    st.header("🎯 Loan Default Risk Predictor")

    # Model selector
    col_header, col_model = st.columns([3, 1])
    with col_header:
        st.markdown("Enter applicant details to get an instant AI-powered risk assessment.")
    with col_model:
        model_choice = st.selectbox(
            "Select Model",
            ["lgbm", "xgb"],
            format_func=lambda x: "🌿 LightGBM" if x == "lgbm" else "⚡ XGBoost"
        )

    # Load selected model
    model, encoders, features, explainer = load_artifacts(model_type=model_choice)

    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Applicant Profile")
        gender        = st.selectbox("Gender", ["M", "F"])
        contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
        income_type   = st.selectbox("Income Type", [
            "Working", "Commercial associate", "Pensioner",
            "State servant", "Unemployed", "Student"
        ])
        education     = st.selectbox("Education", [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ])

    with col2:
        st.subheader("Financial Details")
        income         = st.number_input("Annual Income (₹)",   50000,   10000000, 180000, step=10000)
        credit         = st.number_input("Loan Amount (₹)",     50000,   5000000,  450000, step=10000)
        annuity        = st.number_input("Monthly Annuity (₹)", 1000,    200000,   22500,  step=500)
        goods_price    = st.number_input("Goods Price (₹)",     50000,   5000000,  450000, step=10000)
        age_years      = st.slider("Age (years)",               18,      70,       35)
        employed_years = st.slider("Years Employed",            0,       40,       5)
        family_members = st.slider("Family Members",            1,       10,       2)

    st.subheader("External Credit Scores")
    c1, c2, c3 = st.columns(3)
    with c1:
        ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.6, 0.01)
    with c2:
        ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.7, 0.01)
    with c3:
        ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5, 0.01)

    if st.button("🔍 Assess Risk", type="primary", width='stretch'):
        input_dict = {
            "AMT_INCOME_TOTAL":    income,
            "AMT_CREDIT":          credit,
            "AMT_ANNUITY":         annuity,
            "AMT_GOODS_PRICE":     goods_price,
            "DAYS_BIRTH":          -(age_years * 365),
            "DAYS_EMPLOYED":       -(employed_years * 365),
            "EXT_SOURCE_1":        ext1,
            "EXT_SOURCE_2":        ext2,
            "EXT_SOURCE_3":        ext3,
            "CODE_GENDER":         gender,
            "NAME_CONTRACT_TYPE":  contract_type,
            "NAME_INCOME_TYPE":    income_type,
            "NAME_EDUCATION_TYPE": education,
            "CNT_FAM_MEMBERS":     float(family_members),
        }

        with st.spinner(f"Calculating risk score using {model_choice.upper()}..."):
            result = predict_single(input_dict, model, encoders, features)

        st.divider()

        band  = result["risk_band"]
        color = result["risk_color"]
        score = result["risk_score"]

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:{color}">{score}</div>
                <div class="metric-lbl">Risk Score (0–100)</div>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:{color}">{band}</div>
                <div class="metric-lbl">Risk Band</div>
            </div>""", unsafe_allow_html=True)
        with col_r3:
            prob_pct = round(result['probability'] * 100, 1)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{prob_pct}%</div>
                <div class="metric-lbl">Default Probability</div>
            </div>""", unsafe_allow_html=True)
        with col_r4:
            decision       = "✅ APPROVE" if result["approve"] else "❌ DECLINE"
            decision_color = "#1D9E75"   if result["approve"] else "#E24B4A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:{decision_color};font-size:1.4rem">{decision}</div>
                <div class="metric-lbl">Recommendation</div>
            </div>""", unsafe_allow_html=True)

        st.progress(int(score), text=f"Risk Score: {score}/100")
        st.caption(f"Prediction made using: **{'LightGBM' if model_choice == 'lgbm' else 'XGBoost'}** model")

        if band == "Medium":
            st.warning("⚠️ **Underwriting Policy Alert:** This applicant falls inside the **Medium Risk** band and has been flagged for **Manual Verification & Underwriter Audit** prior to capital disbursement.")

        # Asymmetric Error Cost-Benefit Analysis
        with st.expander("💼 Business Cost-Benefit Analysis"):
            st.markdown(f"""
            Banks face **asymmetric error costs** when making lending decisions. Approving a defaulter (False Negative) is significantly more expensive than declining a creditworthy borrower (False Positive).
            
            **For this requested Loan Amount of ₹{credit:,.2f}:**
            * 🔴 **Default Capital Loss Risk (False Negative Cost):** **₹{credit * 0.60:,.2f}**  
              *(Assumes industry-standard 60% Loss Given Default (LGD) on outstanding principal)*
            * 🟡 **Opportunity Profit Loss Cost (False Positive Cost):** **₹{credit * 0.10:,.2f}**  
              *(Assumes 10% Net Interest Margin (NIM) lost by declining a good borrower)*
            
            **Model Optimization Policy:**
            Because a default is **6x more expensive** than a missed opportunity, our active classifier threshold is swept down to **0.30** (instead of a standard 0.50). This risk-optimal sweep protects an estimated **₹{credit * 0.50:,.2f}** in capital by heavily penalizing default misclassifications, directly aligning this AI platform with bank risk-tolerance guidelines.
            """)

        st.session_state["last_input"]        = input_dict
        st.session_state["last_result"]       = result
        st.session_state["last_model_choice"] = model_choice
        st.info("💡 Go to the **Explainability** tab to see which factors drove this score.")

    # Permanent Credit Policy card visible on tab load
    st.divider()
    st.markdown("""
    ### 💼 Underwriting Policy: Asymmetric Lending Error Costs
    Lending institutions face **asymmetric financial penalties** when credit models make classification errors:
    * **False Negative (FN):** Approving a high-risk applicant who defaults. The bank loses **~60% of the loan principal** *(Loss Given Default - LGD)*.
    * **False Positive (FP):** Declining a low-risk borrower. The bank loses **~10% interest revenue** *(Net Interest Margin - NIM)*.
    
    Because defaults are **6x more expensive** than missed opportunities, our classifiers are calibrated with a risk-optimal threshold of **0.30** (instead of 0.50) to heavily minimize expected capital loss. 
    
    *👉 Enter applicant details and click **🔍 Assess Risk** above to calculate a dynamic cost-benefit analysis for a specific loan amount!*
    """)


# ══════════════════════════════════════════════════════════
# TAB 3 — Explainability
# ══════════════════════════════════════════════════════════
with tab3:
    st.header("🔍 Explainable AI — SHAP Analysis")

    subtab1, subtab2 = st.tabs(["Global Explanation", "Individual Prediction"])

    with subtab1:
        st.subheader("Global Feature Importance")
        st.markdown("Which features most influence default predictions across all applicants.")

        charts = load_shap_charts()
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(charts["summary"]):
                st.markdown("**SHAP Beeswarm — Feature Impact Distribution**")
                st.image(charts["summary"], use_container_width=True)
        with col2:
            if os.path.exists(charts["bar"]):
                st.markdown("**Mean |SHAP| — Average Feature Importance**")
                st.image(charts["bar"], use_container_width=True)

        st.divider()
        col3, col4 = st.columns(2)
        with col3:
            if os.path.exists(charts["roc"]):
                st.markdown("**ROC Curve**")
                st.image(charts["roc"], use_container_width=True)
        with col4:
            if os.path.exists(charts["pr"]):
                st.markdown("**Precision-Recall Curve**")
                st.image(charts["pr"], use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            if os.path.exists(charts["cm"]):
                st.markdown("**Confusion Matrix**")
                st.image(charts["cm"], use_container_width=True)
        with col6:
            if os.path.exists(charts["fi"]):
                st.markdown("**LightGBM Feature Importance**")
                st.image(charts["fi"], use_container_width=True)

    with subtab2:
        st.subheader("Individual Prediction Explanation")

        if "last_input" not in st.session_state:
            st.info("👆 First run a prediction in the **Risk Predictor** tab, then come back here.")
        else:
            input_dict   = st.session_state["last_input"]
            result       = st.session_state["last_result"]
            last_model   = st.session_state.get("last_model_choice", "lgbm")

            st.markdown(f"**Last prediction:** Risk Band = **{result['risk_band']}** | Score = **{result['risk_score']}** | Model = **{last_model.upper()}**")

            with st.spinner("Computing SHAP values for this applicant..."):
                _, encoders_xai, features_xai, explainer_xai = load_artifacts(model_type="lgbm")
                X = preprocess_single(input_dict, encoders_xai, features_xai)
                explanation = explain_single(X, explainer_xai, features_xai)

            st.markdown("#### Top factors driving this prediction")
            for item in explanation[:8]:
                direction = item["direction"]
                shap_val  = item["shap_value"]
                feature   = item["feature"]
                color     = "#E24B4A" if shap_val > 0 else "#1D9E75"
                st.markdown(f"""
                <div style="background:#1e2130;border-radius:8px;padding:10px 14px;margin:5px 0;
                            border-left:3px solid {color}">
                    <span style="color:#c8cad8;font-size:0.9rem"><b>{feature}</b></span>
                    <span style="color:{color};font-size:0.85rem;float:right">{direction} ({shap_val:+.4f})</span>
                </div>
                """, unsafe_allow_html=True)

            try:
                with st.spinner("Generating waterfall plot..."):
                    wf_path = plot_waterfall_single(X, explainer_xai, features_xai)
                if os.path.exists(wf_path):
                    st.markdown("#### Waterfall Plot")
                    st.image(wf_path, use_container_width=True)
            except Exception as e:
                st.warning(f"Waterfall plot skipped: {e}")


# ══════════════════════════════════════════════════════════
# TAB 4 — Talk to Data
# ══════════════════════════════════════════════════════════
with tab4:
    st.header("💬 Talk to Data")
    st.markdown("Ask questions about the dataset in plain English. Powered by **Groq LLaMA-3.3**.")

    init_db()

    st.markdown("**Try a sample question:**")
    cols = st.columns(3)
    for i, q in enumerate(SAMPLE_QUERIES[:6]):
        with cols[i % 3]:
            if st.button(q[:45] + "...", key=f"sample_{i}", width='stretch'):
                st.session_state["chat_input"] = q

    st.divider()

    # 1. Create a placeholder container for all messages (guarantees they stay above the input box)
    chat_container = st.container()

    # Chat history initialization
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # 2. Render previous history INSIDE the chat container
    with chat_container:
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sql"):
                    with st.expander("View SQL Query"):
                        st.code(msg["sql"], language="sql")
                if msg.get("data") is not None:
                    with st.expander("View Raw Data"):
                        st.dataframe(msg["data"].head(10), use_container_width=True)

    # 3. Chat input is rendered below the container
    default_val = st.session_state.pop("chat_input", "")
    user_input = st.chat_input("Ask anything about the credit data...")

    # Determine if a query was submitted
    query = user_input or (default_val if default_val else None)

    if query:
        # ── Rate Limiter Check (30 RPM & 2s Cool-down) ──────
        import time
        current_time = time.time()
        
        if "query_timestamps" not in st.session_state:
            st.session_state["query_timestamps"] = []
            
        st.session_state["query_timestamps"] = [
            t for t in st.session_state["query_timestamps"] if current_time - t < 60
        ]
        
        is_blocked = False
        if st.session_state["query_timestamps"]:
            last_time = st.session_state["query_timestamps"][-1]
            if current_time - last_time < 2.0:
                st.error("⚠️ Slow down! Please wait 2 seconds between consecutive questions.")
                is_blocked = True
                
        if not is_blocked and len(st.session_state["query_timestamps"]) >= 30:
            st.error("⚠️ Rate Limit Exceeded: You have made 30 queries in the last minute. Please wait a moment to protect API tokens.")
            is_blocked = True
            
        if is_blocked:
            time.sleep(1.5)
            st.rerun()
            
        # Passed checks -> record timestamp
        st.session_state["query_timestamps"].append(current_time)
        
        # ── Proceed with Chat Pipeline ──────────────────────
        # 4. We render the new query and spinner INSIDE the container (visually ABOVE the chat_input widget)
        with chat_container:
            st.session_state["chat_history"].append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = ask(query)

                if result["error"]:
                    st.error(f"Query failed: {result['error']}")
                    response_msg = f"Sorry, I couldn't answer that: {result['error']}"
                else:
                    st.markdown(result["answer"])
                    if result["sql"]:
                        with st.expander("View SQL Query"):
                            st.code(result["sql"], language="sql")
                    if result["data"] is not None and len(result["data"]) > 0:
                        with st.expander("View Raw Data"):
                            st.dataframe(result["data"].head(10), use_container_width=True)
                    response_msg = result["answer"]

        # Save to history
        st.session_state["chat_history"].append({
            "role":    "assistant",
            "content": response_msg,
            "sql":     result.get("sql"),
            "data":    result.get("data"),
        })
        
        # 5. Rerun to refresh the full history inside the container cleanly
        st.rerun()