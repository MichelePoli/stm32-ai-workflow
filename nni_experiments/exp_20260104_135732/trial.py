
import sys
import os
import nni
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- AUTO-CONFIGURE CUDA PATH (Robust) ---
# Must be done BEFORE loading TensorFlow
if 'LD_LIBRARY_PATH' not in os.environ:
    conda_lib_path = os.path.join(sys.prefix, 'lib')
    if os.path.exists(conda_lib_path):
        os.environ['LD_LIBRARY_PATH'] = f"{conda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        print(f"[TRIAL] 🔧 Added Conda lib to LD_LIBRARY_PATH: {conda_lib_path}")
        
        # Self-restart to apply env vars to dynamic linker
        if 'RESTARTED_WITH_LD' not in os.environ:
             print("[TRIAL] 🔄 Restarting script to apply environment...")
             os.environ['RESTARTED_WITH_LD'] = 'true'
             try:
                 os.execv(sys.executable, [sys.executable] + sys.argv)
             except Exception as e:
                 print(f"[TRIAL] ⚠️ Restart failed: {e}")

# GPU Memory Growth (Prevent OOM)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[TRIAL] 🎮 GPU initialized: {len(gpus)} devices")
    except RuntimeError as e:
        print(e)

# Check mode
IS_RETRAIN = os.environ.get('RETRAIN_MODE', 'false').lower() == 'true'

if IS_RETRAIN:
    print("[TRIAL] 🔄 Retraining mode activated!")
    params = json.loads(os.environ.get('NNI_PARAMS', '{}'))
else:
    # Standard NNI Mode
    params = nni.get_next_parameter()

# Get hyperparameters
lr = params.get('learning_rate', 0.001)
batch_size = params.get('batch_size', 32)
optimizer_name = params.get('optimizer', 'Adam')

# Load data with MMAP to save RAM
print("[TRIAL] 📂 Loading data (Memory Mapped)...")
x_train = np.load(f'/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/x_train.npy', mmap_mode='r')
y_train = np.load(f'/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/y_train.npy', mmap_mode='r')
x_test = np.load(f'/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/x_test.npy', mmap_mode='r')
y_test = np.load(f'/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/y_test.npy', mmap_mode='r')

# --- DATASET SUBSETTING (Memory Optimization - 50%) ---
# We take only a subset of the data to reduce RAM usage during training
subset_ratio = 0.5
num_samples = int(len(x_train) * subset_ratio)

# Simple slicing for mmap efficiency (avoid random access on risk of thrashing if not careful, 
# but linear slice is safe and effective for memory)
x_train = x_train[:num_samples]
y_train = y_train[:num_samples]

print(f"[TRIAL] ✂️ Subsetting dataset: {len(x_train)} samples ({subset_ratio*100}%)")

# --- DATASET FIX: Ensure One-Hot Labels (Categorical) ---
# Check if labels are sparse (1D or 2D with last dim 1)
if len(y_train.shape) == 1 or (len(y_train.shape) == 2 and y_train.shape[-1] == 1):
    num_classes = len(np.unique(y_train))
    print(f"[TRIAL] Sparse labels detected. Classes: {num_classes}")
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
else:
    num_classes = y_train.shape[-1]
    print(f"[TRIAL] Categorical labels detected. Classes: {num_classes}")

print(f"[TRIAL] Dataset classes: {num_classes}")

# Load model
model = keras.models.load_model('/tmp/customized_model.keras')

# --- MODEL FIX: Adaptive Output Layer ---
if model.output_shape[-1] != num_classes:
    print(f"[TRIAL] ⚠️ Class Mismatch: Model={model.output_shape[-1]}, Data={num_classes}")
    print("[TRIAL] 🔧 Replacing final layer...")
    
    # Simple replacement for Sequential/Functional (assumes last layer is Dense)
    x = model.layers[-2].output
    output = keras.layers.Dense(num_classes, activation='softmax', name='adaptive_output')(x)
    model = keras.Model(inputs=model.input, outputs=output)
    print(f"[TRIAL] ✓ New output shape: {model.output_shape}")

# --- PARAMETER APPLICATION ---
# Optimizer Selection (Adam vs SGD)
if optimizer_name.lower() == 'sgd':
    opt = keras.optimizers.SGD(learning_rate=lr)
else:
    opt = keras.optimizers.Adam(learning_rate=lr)

# --- SHAPE FIX: Resize images if needed ---
expected_shape = model.input_shape[1:3]  # (H, W)
print(f"Model expects: {expected_shape}, Data has: {x_train.shape[1:3]}")

def preprocess(x, y):
    # 1. Cast to float32 (CRITICAL for compatibility)
    x = tf.cast(x, tf.float32)
    # 2. Normalize to [0...1]
    x = x / 255.0
    # 3. Resize to expected shape
    x = tf.image.resize(x, expected_shape)
    return x, y

# Datasets (Removed caching of resized images to save RAM)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(1000).map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# FINAL COMPILE (Generate ONLY this block)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
print(f"[TRIAL] 🚀 Starting training (First epoch may be slow due to resizing/caching)...")
history = model.fit(train_ds, 
                    validation_data=val_ds,
                    epochs=3, verbose=1)

# Report or Save
if not IS_RETRAIN:
    val_accuracy = history.history['val_accuracy'][-1]
    print(f"[TRIAL] Validation accuracy: {val_accuracy:.4f}")
    nni.report_final_result(val_accuracy)
else:
    model.save('best_model.h5')
    print(f"[TRIAL] ✅ Best model saved.")
