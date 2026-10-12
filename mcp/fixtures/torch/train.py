import argparse
import json
import os
from pathlib import Path
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    for key in ("train", "validation", "metadata", "models", "output", "seed", "config"):
        parser.add_argument("--" + key)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    meta = json.loads(Path(args.metadata).read_text())
    device = os.environ.get("HELIX_DEVICE", config.get("device", "cpu"))
    torch.manual_seed(int(args.seed))
    frame = pd.read_csv(args.train, dtype=str)
    x = torch.tensor(frame[meta["features"]].astype(float).values, dtype=torch.float32)
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-6)
    x = ((x - mean) / scale).to(device)
    y = torch.tensor([meta["classes"].index(label) for label in frame[meta["target"]]], device=device)
    valid = torch.tensor(pd.read_csv(args.validation)[meta["features"]].values, dtype=torch.float32)
    valid = ((valid - mean) / scale).to(device)
    model = torch.nn.Linear(x.shape[1], len(meta["classes"])).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0)
    history = []
    for step in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        history.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    with torch.no_grad():
        predictions = torch.softmax(model(valid), dim=1).cpu().tolist()
    output = Path(args.output).parent
    Path(args.output).write_text(json.dumps(predictions))
    (output / "training_history.json").write_text(json.dumps(history))
    (output / "device_check.json").write_text(json.dumps({"device": str(next(model.parameters()).device), "tensorDevice": str(x.device),
        "gpuBytes": torch.cuda.max_memory_allocated() if device == "cuda" else torch.mps.current_allocated_memory() if device == "mps" else 0}))
    if config["exportModel"]:
        folder = output / "model"
        folder.mkdir()
        torch.save({"weights": model.cpu().state_dict(), "mean": mean, "scale": scale}, folder / "weights.pt")
        (folder / "model_manifest.json").write_text(json.dumps({"format": "pytorch", "files": ["weights.pt"]}))


if __name__ == "__main__":
    main()
