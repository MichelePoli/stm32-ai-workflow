import os
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)


def generate_nni_experiment(
    model_info: Dict[str, Any],
    dataset_info: Dict[str, Any],
    optimization_goal: str = "Maximize validation accuracy and minimize latency",
    output_dir: str = "./nni_experiment",
    num_ctx: int = 8192,
) -> Dict[str, str]:
    """
    Generates NNI experiment scripts (manager.py, trial.py) using Python templates.

    Previously this function called an LLM to generate the scripts, but the prompt
    exceeded max_model_len=2048 of gpt-oss-20b, causing vLLM to silently return an
    empty string. Since manager.py / trial.py have a deterministic structure and only
    the dataset/model PATHS change per experiment, we now use Python f-string templates
    with direct path injection — faster, 100% reliable, no LLM call needed.

    Args:
        model_info:  Dictionary containing model architecture details (name, path, …).
        dataset_info: Dictionary containing dataset details (path, num_classes, …).
        optimization_goal: Objective description (informational only).
        output_dir: Directory to save generated files.
        num_ctx: (kept for API compatibility, unused).

    Returns:
        Dict mapping filenames to their generated content.
    """
    logger.info(f"🤖 Generating NNI experiment for model: {model_info.get('name', 'Unknown')}")

    data_path  = dataset_info.get("path", "")
    model_path = model_info.get("path", "")

    # ── manager.py ────────────────────────────────────────────────────────────
    manager_content = f"""\
import nni
import os
import sys
import json
import subprocess
from nni.experiment import Experiment

# Fix for Conda environments that lack CUDA libs in LD_LIBRARY_PATH
conda_lib_path = os.path.join(sys.prefix, 'lib')
if os.path.exists(conda_lib_path):
    os.environ['LD_LIBRARY_PATH'] = conda_lib_path + ':' + os.environ.get('LD_LIBRARY_PATH', '')

current_dir = os.path.dirname(os.path.abspath(__file__))

search_space = {{
    'learning_rate':   {{'_type': 'choice', '_value': [0.001, 0.0005, 0.0001]}},
    'batch_size':      {{'_type': 'choice', '_value': [16, 32, 64]}},
    'optimizer':       {{'_type': 'choice', '_value': ['Adam']}},
    'freeze_backbone': {{'_type': 'choice', '_value': [True, False]}},
}}

experiment = Experiment('local')
experiment.config.trial_command = f'{{sys.executable}} trial.py'
experiment.config.trial_code_directory = current_dir
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Random'
experiment.config.tuner.class_args = {{'optimize_mode': 'maximize'}}
experiment.config.max_trial_number = 8
experiment.config.trial_concurrency = 2
experiment.config.training_service.use_active_gpu = True

try:
    import socket

    def find_free_port(start=8080, n=20):
        for p in range(start, start + n):
            with socket.socket() as s:
                if s.connect_ex(('localhost', p)) != 0:
                    return p
        return start

    port = int(os.environ.get('NNI_PORT', '0')) or find_free_port()
    print(f'[NNI] 🌐 Starting experiment on port {{port}}  →  http://localhost:{{port}}')
    experiment.run(port=port, wait_completion=True)
    print('[NNI] Experiment completed')

    trials = experiment.list_trial_jobs()
    valid  = [t for t in trials if t.status == 'SUCCEEDED' and t.finalMetricData]
    if not valid:
        raise Exception('No successful trials')

    best = max(valid, key=lambda t: float(t.finalMetricData[0].data))
    hp = best.hyperParameters
    if isinstance(hp, list):
        hp = hp[-1]
    if hasattr(hp, 'parameters'):
        best_params = hp.parameters
    elif isinstance(hp, dict):
        best_params = hp.get('parameters', hp)
    else:
        best_params = hp

    print(f'[NNI] Best params: {{best_params}}')
    env = os.environ.copy()
    env['RETRAIN_MODE'] = 'true'
    env['NNI_PARAMS']   = json.dumps(best_params)
    subprocess.run([sys.executable, 'trial.py'], cwd=current_dir, env=env, check=True)
    print('[NNI] Best model saved.')

except Exception as e:
    print(f'[NNI] Error: {{e}}')
    import traceback
    traceback.print_exc()

finally:
    experiment.stop()
"""

    # ── trial.py ──────────────────────────────────────────────────────────────
    trial_content = f"""\
import sys
import os

# TF_USE_LEGACY_KERAS=True is set in the Docker container environment, but
# tf_keras is not installed in the stm32 conda env. Unsetting it here prevents
# the "you must install tf_keras" crash that silently kills every NNI trial.
os.environ.pop('TF_USE_LEGACY_KERAS', None)

import json
import nni
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Fix LD_LIBRARY_PATH for Conda envs
if 'LD_LIBRARY_PATH' not in os.environ:
    lp = os.path.join(sys.prefix, 'lib')
    if os.path.exists(lp) and 'RESTARTED_WITH_LD' not in os.environ:
        os.environ['LD_LIBRARY_PATH'] = lp
        os.environ['RESTARTED_WITH_LD'] = 'true'
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            pass

# GPU memory growth
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError:
        pass

IS_RETRAIN = os.environ.get('RETRAIN_MODE', 'false').lower() == 'true'
params = json.loads(os.environ.get('NNI_PARAMS', '{{}}')) if IS_RETRAIN else nni.get_next_parameter()

lr          = params.get('learning_rate', 0.001)
batch_size  = params.get('batch_size', 32)
opt_name    = params.get('optimizer', 'Adam')
freeze_bb   = params.get('freeze_backbone', False)

# ── Load data (50% subset to save RAM) ───────────────────────────────────────
DATA    = r'{data_path}'
x_train = np.load(DATA + '/x_train.npy', mmap_mode='r')
y_train = np.load(DATA + '/y_train.npy', mmap_mode='r')
x_test  = np.load(DATA + '/x_test.npy',  mmap_mode='r')
y_test  = np.load(DATA + '/y_test.npy',  mmap_mode='r')

n = len(x_train) // 2
x_train = x_train[:n]
y_train = y_train[:n]

# One-hot encode sparse labels
if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[-1] == 1):
    nc      = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, nc)
    y_test  = keras.utils.to_categorical(y_test,  nc)
else:
    nc = y_train.shape[-1]

# ── Load model ────────────────────────────────────────────────────────────────
model = keras.models.load_model(r'{model_path}', compile=False)

# Adapt output head if needed
if model.output_shape[-1] != nc:
    x   = model.layers[-2].output
    out = keras.layers.Dense(nc, activation='softmax', name='out')(x)
    model = keras.Model(inputs=model.input, outputs=out)

if freeze_bb:
    for layer in model.layers[:-5]:
        layer.trainable = False

opt = (keras.optimizers.SGD(lr)
       if opt_name.lower() == 'sgd'
       else keras.optimizers.Adam(lr))

hw, hh = model.input_shape[1], model.input_shape[2]


def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.image.resize(x, [hw, hh])
    return x, y


aug = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1000)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .map(lambda x, y: (aug(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)

if IS_RETRAIN:
    model.save('best_model.h5')
    print('[TRIAL] Best model saved.')
else:
    nni.report_final_result(history.history['val_accuracy'][-1])
"""

    files = {
        "manager.py": manager_content,
        "trial.py":   trial_content,
    }

    # ── Save to disk ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    for filename, content in files.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        logger.info(f"   ✓ Written: {path}")

    logger.info("✅ NNI experiment files generated via template (no LLM call)")
    return files


if __name__ == "__main__":
    # Test stub
    model_dummy = {"name": "TestModel", "path": "/tmp/model.h5", "input_shape": "(32,32,3)", "n_layers": 10}
    data_dummy  = {"path": "/tmp/data", "num_classes": 10}
    result = generate_nni_experiment(model_dummy, data_dummy, output_dir="/tmp/nni_test")
    print(f"Generated: {list(result.keys())}")
