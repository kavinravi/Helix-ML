"""Activate wheel startup hooks in a candidate directory, then execute its Python command."""
import os
import runpy
import site
import sys

# PYTHONPATH supplies imports but does not process .pth files (e.g. numba-cuda).
site.addsitedir(os.environ["HELIX_PACKAGES"])
sys.argv = sys.argv[1:]
if sys.argv[0] == "-c":
    code = sys.argv.pop(1)
    exec(compile(code, "<string>", "exec"), {"__name__": "__main__", "__builtins__": __builtins__})
elif sys.argv[0] == "-m":
    sys.argv.pop(0)
    runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
else:
    runpy.run_path(sys.argv[0], run_name="__main__")
