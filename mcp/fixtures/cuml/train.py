import argparse
import json
import os
from pathlib import Path
import cupy as cp
from cuml.cluster import KMeans
from cuml.decomposition import PCA
import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    for key in ('train', 'validation', 'metadata', 'models', 'output', 'seed', 'config'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    assert os.environ['HELIX_DEVICE'] == 'cuda'
    config = json.loads(Path(args.config).read_text())
    meta = json.loads(Path(args.metadata).read_text())
    train = pd.read_csv(args.train)[meta['features']].to_numpy(dtype=np.float32)
    valid = pd.read_csv(args.validation)[meta['features']].to_numpy(dtype=np.float32)
    x, v = cp.asarray(train), cp.asarray(valid)
    model = KMeans(n_clusters=2, random_state=int(args.seed), output_type='numpy', n_init=1) if config['learning'] == 'clustering' else PCA(n_components=config['dimensions'], output_type='numpy')
    model.fit(x)
    pred = model.predict(v) if config['learning'] == 'clustering' else model.transform(v)
    cp.cuda.get_current_stream().synchronize()
    Path(args.output).write_text(json.dumps(pred.tolist()))
    output = Path(args.output).parent
    (output / 'device_check.json').write_text(json.dumps({'device': 'cuda', 'gpuBytes': x.nbytes + v.nbytes, 'gpu': x.device.id}))
    if config['exportModel']:
        folder = output / 'model'
        folder.mkdir()
        joblib.dump(model, folder / 'model.joblib')
        (folder / 'model_manifest.json').write_text(json.dumps({'format': 'joblib', 'files': ['model.joblib']}))


if __name__ == '__main__':
    main()
