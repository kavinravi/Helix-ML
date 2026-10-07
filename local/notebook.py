"""Execute the exported notebook with a real kernel inside the training sandbox."""
import os
import sys
import nbformat
from nbclient import NotebookClient

os.environ.update(HELIX_SOURCE="/code", HELIX_DATA="/data", HELIX_OUTPUT="/work", HELIX_MODELS="/models", HELIX_SEED=sys.argv[1], JUPYTER_RUNTIME_DIR="/tmp/jupyter")
notebook = nbformat.read("/code/solution.ipynb", as_version=4)
NotebookClient(notebook, timeout=None, kernel_name="python3", resources={"metadata": {"path": "/work"}}).execute()
