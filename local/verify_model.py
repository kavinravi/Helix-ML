"""Verify that an export can be opened by its declared format's native loader."""
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
suffixes = {"joblib": ".joblib", "pickle": ".pkl", "pytorch": ".pt", "torchscript": ".pt", "keras": ".keras", "savedmodel": "saved_model.pb", "onnx": ".onnx"}
paths = [manifest_path.parent / name for name in manifest["files"] if name.endswith(suffixes[manifest["format"]])]
for path in paths:
    if manifest["format"] == "joblib":
        import joblib
        joblib.load(path)
    elif manifest["format"] == "pickle":
        import pickle
        with path.open("rb") as stream:
            pickle.load(stream)
    elif manifest["format"] == "pytorch":
        import torch
        result = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(result, dict) or not result:
            raise ValueError("PyTorch export must contain a nonempty state_dict")
    elif manifest["format"] == "torchscript":
        import torch
        torch.jit.load(str(path), map_location="cpu")
    elif manifest["format"] == "keras":
        import keras
        keras.models.load_model(path, compile=False, safe_mode=True)
    elif manifest["format"] == "savedmodel":
        import tensorflow as tf
        tf.saved_model.load(str(path.parent))
    elif manifest["format"] == "onnx":
        import onnx
        onnx.checker.check_model(str(path))
print("Native format loader accepted the exported model.")
