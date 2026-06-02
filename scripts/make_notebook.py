import nbformat as nbf

with open('notebooks/eda.py', 'r') as f:
    code = f.read()

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("""# Credit Risk EDA
Exploratory Data Analysis on the Home Credit Default Risk Dataset.

This notebook covers:
- Dataset summary and data quality
- Class imbalance analysis
- 6 key business insights
- Feature correlation analysis
- Missing value analysis
"""),
    nbf.v4.new_code_cell(code)
]

nbf.write(nb, open('notebooks/eda.ipynb', 'w'))
print("Done — notebooks/eda.ipynb created successfully")