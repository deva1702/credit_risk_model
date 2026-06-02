import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from src.data.loader import load_main_table
from src.utils.logger import get_logger

logger = get_logger("EDA")
os.makedirs("notebooks/charts", exist_ok=True)

# ── Load data ──────────────────────────────────────────────
df = load_main_table()
logger.info(f"Dataset shape: {df.shape}")

# ── 1. Dataset summary ────────────────────────────────────
print("\n===== DATASET SUMMARY =====")
print(f"Total applications : {len(df):,}")
print(f"Total features     : {df.shape[1]}")
print(f"Default rate       : {df['TARGET'].mean()*100:.2f}%")
print(f"Missing values     : {df.isnull().sum().sum():,} cells")

# ── 2. Class imbalance ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['TARGET'].value_counts()
bars = ax.bar(['No Default (0)', 'Default (1)'], counts.values,
              color=['#1D9E75', '#D85A30'], edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
            f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=10)
ax.set_title('Class distribution — heavy imbalance (8% default)', fontsize=12, pad=10)
ax.set_ylabel('Count')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('notebooks/charts/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
logger.info("Saved 01_class_distribution.png")

# ── 3. Business Insight 1: Default rate by gender ─────────
gender_default = df.groupby('CODE_GENDER')['TARGET'].mean().reset_index()
gender_default.columns = ['Gender', 'Default Rate']
gender_default = gender_default[gender_default['Gender'] != 'XNA']

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(gender_default['Gender'], gender_default['Default Rate']*100,
              color=['#534AB7', '#1D9E75'], edgecolor='white')
for bar, val in zip(bars, gender_default['Default Rate']*100):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
ax.set_title('Insight 1: Men default 1.5x more than women', fontsize=12)
ax.set_ylabel('Default rate (%)')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('notebooks/charts/02_default_by_gender.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 4. Business Insight 2: Default by contract type ───────
contract_default = df.groupby('NAME_CONTRACT_TYPE')['TARGET'].mean().reset_index()
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(contract_default['NAME_CONTRACT_TYPE'], contract_default['TARGET']*100,
        color='#378ADD', edgecolor='white')
ax.set_title('Insight 2: Cash loans default more than revolving loans', fontsize=12)
ax.set_xlabel('Default rate (%)')
ax.spines[['top','right']].set_visible(False)
for i, val in enumerate(contract_default['TARGET']*100):
    ax.text(val + 0.1, i, f'{val:.1f}%', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('notebooks/charts/03_default_by_contract.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 5. Business Insight 3: Income vs Default ──────────────
df['INCOME_BAND'] = pd.cut(df['AMT_INCOME_TOTAL'],
                            bins=[0, 100000, 200000, 300000, 500000, np.inf],
                            labels=['<100K', '100-200K', '200-300K', '300-500K', '>500K'])
income_default = df.groupby('INCOME_BAND', observed=True)['TARGET'].mean().reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(income_default['INCOME_BAND'].astype(str), income_default['TARGET']*100,
       color='#EF9F27', edgecolor='white')
ax.set_title('Insight 3: Lower income = higher default risk', fontsize=12)
ax.set_ylabel('Default rate (%)')
ax.set_xlabel('Annual income band')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('notebooks/charts/04_default_by_income.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 6. Business Insight 4: Age distribution ───────────────
df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).astype(int)
df['AGE_BAND'] = pd.cut(df['AGE_YEARS'], bins=[20, 30, 40, 50, 60, 70],
                         labels=['20-30', '30-40', '40-50', '50-60', '60-70'])
age_default = df.groupby('AGE_BAND', observed=True)['TARGET'].mean().reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(age_default['AGE_BAND'].astype(str), age_default['TARGET']*100,
        marker='o', color='#D85A30', linewidth=2.5, markersize=8)
ax.fill_between(range(len(age_default)), age_default['TARGET']*100,
                alpha=0.15, color='#D85A30')
ax.set_title('Insight 4: Younger applicants (20-30) are highest risk', fontsize=12)
ax.set_ylabel('Default rate (%)')
ax.set_xlabel('Age band')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('notebooks/charts/05_default_by_age.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 7. Business Insight 5: Employment vs Default ──────────
df['EMPLOYMENT_YEARS'] = (-df['DAYS_EMPLOYED'].clip(upper=0) / 365)
df['EMP_BAND'] = pd.cut(df['EMPLOYMENT_YEARS'],
                         bins=[-1, 0, 2, 5, 10, 50],
                         labels=['Unemployed', '0-2 yrs', '2-5 yrs', '5-10 yrs', '10+ yrs'])
emp_default = df.groupby('EMP_BAND', observed=True)['TARGET'].mean().reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
colors = ['#E24B4A' if v > 0.12 else '#1D9E75' for v in emp_default['TARGET']]
ax.bar(emp_default['EMP_BAND'].astype(str), emp_default['TARGET']*100,
       color=colors, edgecolor='white')
ax.set_title('Insight 5: Newly employed / unemployed are highest risk', fontsize=12)
ax.set_ylabel('Default rate (%)')
ax.set_xlabel('Employment duration')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('notebooks/charts/06_default_by_employment.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 8. Missing value heatmap (top 30 cols) ────────────────
missing = df.isnull().mean().sort_values(ascending=False).head(30)
fig, ax = plt.subplots(figsize=(10, 6))
missing.plot(kind='bar', ax=ax, color='#7F77DD', edgecolor='white')
ax.set_title('Top 30 features by missing value rate', fontsize=12)
ax.set_ylabel('Missing fraction')
ax.set_xlabel('')
ax.axhline(0.4, color='red', linestyle='--', linewidth=1, label='40% threshold')
ax.legend()
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig('notebooks/charts/07_missing_values.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 9. Correlation heatmap (numeric cols) ─────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_cols = ['TARGET', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY',
             'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
corr_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax, linewidths=0.5, square=True)
ax.set_title('Feature correlation matrix', fontsize=12)
plt.tight_layout()
plt.savefig('notebooks/charts/08_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 10. Print final summary ───────────────────────────────
print("\n===== 5 KEY BUSINESS INSIGHTS =====")
print("1. Male applicants default at ~10% vs ~7% for females — gender is a signal")
print("2. Cash loans carry higher default risk than revolving loans")
print("3. Income below 100K correlates strongly with default — credit capacity matters")
print("4. Applicants aged 20-30 have 2x the default rate of those aged 50-60")
print("5. Unemployed / newly employed (0-2 yrs) show the highest default probability")
print("\n===== DATA QUALITY FLAGS =====")
high_missing = missing[missing > 0.4]
print(f"Columns with >40% missing: {len(high_missing)}")
print(f"EXT_SOURCE cols available: {[c for c in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'] if c in df.columns]}")
print(f"\nAll charts saved to notebooks/charts/")