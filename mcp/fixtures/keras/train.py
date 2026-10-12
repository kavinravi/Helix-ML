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
    for key in ('train', 'validation', 'metadata', 'models', 'output', 'seed', 'config'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    meta = json.loads(Path(args.metadata).read_text())
    requested = os.environ.get('HELIX_DEVICE', 'cpu')
    if requested == 'cuda':
        if not tf.config.list_physical_devices('GPU'):
            raise ValueError('TensorFlow CUDA is unavailable in this environment')
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    elif requested != 'cpu':
        raise ValueError('This TensorFlow implementation supports CPU/CUDA, not MPS')
    tf.config.set_soft_device_placement(False)
    keras.utils.set_random_seed(int(args.seed))
    frame = pd.read_csv(args.train, dtype=str)
    x = frame[meta['features']].to_numpy(dtype=np.float32)
    y = np.array([meta['classes'].index(label) for label in frame[meta['target']]])
    valid = pd.read_csv(args.validation)[meta['features']].to_numpy(dtype=np.float32)
    with tf.device('/GPU:0' if requested == 'cuda' else '/CPU:0'):
        model = keras.Sequential([keras.layers.Input(shape=(x.shape[1],)), keras.layers.Dense(len(meta['classes']), activation='softmax')])
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss='sparse_categorical_crossentropy', jit_compile=False)
        history = []
        for step in range(30):
            loss = model.train_on_batch(x, y)
            history.append({'step': step + 1, 'loss': float(loss)})
        pred = model(valid, training=False)
        proof = {'device': pred.device, 'weightDevice': model.weights[0].value.device}
        predictions = pred.numpy().tolist()
    output = Path(args.output).parent
    Path(args.output).write_text(json.dumps(predictions))
    (output / 'device_check.json').write_text(json.dumps(proof))
    (output / 'training_history.json').write_text(json.dumps(history))
    if config['exportModel']:
        folder = output / 'model'
        folder.mkdir()
        fmt = config['exportFormat']
        if fmt == 'savedmodel':
            model.export(str(folder / 'saved_model'))
            files = [p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()]
        else:
            model.save(folder / 'model.keras')
            files = ['model.keras']
        (folder / 'model_manifest.json').write_text(json.dumps({'format': fmt, 'files': files}))


if __name__ == '__main__':
    main()
