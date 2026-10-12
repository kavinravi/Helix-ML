import argparse
import json
import os
from pathlib import Path
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    for key in ("input", "metadata", "config", "model-dir", "models", "output"):
        parser.add_argument("--" + key)
    args = parser.parse_args()
    meta = json.loads(Path(args.metadata).read_text())
    config = json.loads(Path(args.config).read_text())
    device = os.environ.get("HELIX_DEVICE", config.get("device", "cpu"))
    saved = torch.load(Path(args.model_dir) / "weights.pt", map_location="cpu", weights_only=True)
    model = torch.nn.Linear(len(meta["features"]), len(meta["classes"]))
    model.load_state_dict(saved["weights"])
    model.to(device).eval()
    x = torch.tensor(pd.read_csv(args.input)[meta["features"]].values, dtype=torch.float32)
    x = ((x - saved["mean"]) / saved["scale"]).to(device)
    with torch.no_grad():
        predictions = torch.softmax(model(x), dim=1).cpu().tolist()
    Path(args.output).write_text(json.dumps(predictions))


if __name__ == "__main__":
    main()
