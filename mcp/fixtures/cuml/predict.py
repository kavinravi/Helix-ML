import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import cupy as cp


def main():
    parser = argparse.ArgumentParser()
    for key in ('input', 'metadata', 'model-dir', 'models', 'output', 'config'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    meta = json.loads(Path(args.metadata).read_text())
    model = joblib.load(Path(args.model_dir) / 'model.joblib')
    x = cp.asarray(pd.read_csv(args.input)[meta['features']].to_numpy(dtype=np.float32))
    pred = model.predict(x) if config['learning'] == 'clustering' else model.transform(x)
    Path(args.output).write_text(json.dumps(pred.tolist()))


if __name__ == '__main__':
    main()
