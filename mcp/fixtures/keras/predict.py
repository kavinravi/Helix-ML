import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser()
    for key in ('input', 'metadata', 'model-dir', 'models', 'output', 'config'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    meta = json.loads(Path(args.metadata).read_text())
    requested = os.environ.get('HELIX_DEVICE', config.get('device', 'cpu'))
    if requested == 'cuda' and not tf.config.list_physical_devices('GPU'):
        raise ValueError('TensorFlow CUDA unavailable')
    tf.config.set_soft_device_placement(False)
    x = pd.read_csv(args.input)[meta['features']].to_numpy(dtype=np.float32)
    with tf.device('/GPU:0' if requested == 'cuda' else '/CPU:0'):
        if config['exportFormat'] == 'savedmodel':
            model = tf.saved_model.load(str(Path(args.model_dir) / 'saved_model'))
            pred = model.serve(tf.constant(x))
        else:
            model = keras.models.load_model(Path(args.model_dir) / 'model.keras', compile=False, safe_mode=True)
            pred = model(x, training=False)
    Path(args.output).write_text(json.dumps(pred.numpy().tolist()))


if __name__ == '__main__':
    main()
