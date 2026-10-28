"""
Execute churn_project.ipynb top-to-bottom and save results with all cell outputs
"""

import os
import sys
import nbformat
from nbclient import NotebookClient

notebook_path = os.path.join("notebooks", "churn_project.ipynb")
print(f"Reading notebook: {notebook_path}")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": "."}})

print("Executing all cells in notebook...")
try:
    client.execute()
    print("Notebook executed successfully!")
except Exception as e:
    print(f"Error during execution: {e}", file=sys.stderr)
    raise

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Saved executed notebook to: {notebook_path}")
