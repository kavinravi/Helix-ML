"""Execute the exported notebook with a real kernel inside the training sandbox."""
import os
import sys
import nbformat
from nbclient import NotebookClient

for key, default in dict(HELIX_SOURCE="/code", HELIX_DATA="/data", HELIX_OUTPUT="/work", HELIX_MODELS="/models", JUPYTER_RUNTIME_DIR="/tmp/jupyter").items():
    os.environ.setdefault(key, default)
os.environ["HELIX_SEED"] = sys.argv[1]
notebook = nbformat.read(os.path.join(os.environ["HELIX_SOURCE"], "solution.ipynb"), as_version=4)
NotebookClient(notebook, timeout=None, kernel_name="python3", resources={"metadata": {"path": os.environ["HELIX_OUTPUT"]}}).execute()
