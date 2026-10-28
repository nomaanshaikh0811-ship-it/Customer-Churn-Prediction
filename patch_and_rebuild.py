"""
Patch build_and_run_notebook.py to ensure path portability and regenerate churn_project.ipynb
"""

import re

with open("build_and_run_notebook.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace setup block in cell 1
old_init = """RANDOM_STATE = 42
os.makedirs("../outputs/figures", exist_ok=True)
os.makedirs("../models", exist_ok=True)
print("Environment initialized successfully. Random state fixed to 42.")"""

new_init = """RANDOM_STATE = 42

# Ensure robust path resolution regardless of execution working directory
current_dir = os.getcwd()
if os.path.basename(current_dir) == 'notebooks':
    BASE_DIR = os.path.abspath('..')
else:
    BASE_DIR = os.path.abspath('.')

DATA_PATH = os.path.join(BASE_DIR, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
FIG_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Environment initialized. Project root: {BASE_DIR}")
print(f"Figures directory: {FIG_DIR}")
print(f"Models directory:  {MODEL_DIR}")"""

content = content.replace(old_init, new_init)

# Replace data_path loading
content = content.replace('data_path = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"', 'data_path = DATA_PATH')

# Replace savefig calls
content = re.sub(r'plt\.savefig\("\.\./outputs/figures/([^"]+)"', r'plt.savefig(os.path.join(FIG_DIR, "\1")', content)

# Replace model paths
content = content.replace('best_model_path = "../models/best_churn_model.joblib"', 'best_model_path = os.path.join(MODEL_DIR, "best_churn_model.joblib")')
content = content.replace('cluster_model_path = "../models/retention_cluster_model.joblib"', 'cluster_model_path = os.path.join(MODEL_DIR, "retention_cluster_model.joblib")')

with open("build_and_run_notebook.py", "w", encoding="utf-8") as f:
    f.write(content)

print("build_and_run_notebook.py successfully patched!")
