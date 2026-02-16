
import sys
import os
import json
import nni
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- AUTO-CONFIGURE CUDA PATH (Robust) ---
if 'LD_LIBRARY_PATH' not in os.environ:
    conda_lib_path = os.path.join(sys.prefix, 'lib')
    if os.path.exists(conda_lib_path):
        os.environ['LD_LIBRARY_PATH'] = f"{conda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        print(f"[TRIAL] 🔧 Added Conda lib to LD_LIBRARY_PATH: {conda_lib_path}")
        if 'RESTARTED_WITH_LD' not in os.environ:
            print("[TRIAL] 🔄 Restarting script to apply environment...")
            os.environ['RESTARTED_WITH_LD'] = 'true'
            os.execv(sys.executable, [sys.executable] + sys.argv)

# GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[TRIAL] 🎮 GPU initialized: {len(gpus)} devices")
    except RuntimeError as e:
        print(e)

IS_RETRAIN = os.environ.get('RETRAIN_MODE', 'false').lower() == 'true'

if IS_RETRAIN:
    print("[TRIAL] 🔄 Retraining mode activated!")
    params = json.loads(os.environ.get('NNI_PARAMS', '{}'))
else:
    params = nni.get_next_parameter()

lr = params.get('learning_rate', 0.001)
batch_size = params.get('batch_size', 32)
optimizer_name = params.get('optimizer', 'Adam')
freeze_backbone = params.get('freeze_backbone', False)

print("[TRIAL] 📂 Loading data (Memory Mapped)...")
x_train = np.load('/home/mrusso/stm32-ai-workflow/STM32CubeMX/data/real_datasets/cifar10/x_train.npy', mmap_mode='r')
y_train = np.load('/home/mrusso/stm32-ai-workflow/STM32CubeMX/data/real_datasets/cifar10/y_train.npy', mmap_mode='r')
x_test = np.load('/home/mrusso/stm32-ai-workflow/STM32CubeMX/data/real_datasets/cifar10/x_test.npy', mmap_mode='r')
y_test = np.load('/home/mrusso/stm32-ai-workflow/STM32CubeMX/data/real_datasets/cifar10/y_test.npy', mmap_mode='r')

subset_ratio = 0.5
num_samples = int(len(x_train) * subset_ratio)
x_train = x_train[:num_samples]
y_train = y_train[:num_samples]
print(f"[TRIAL] ✂️ Subsetting dataset: {len(x_train)} samples ({subset_ratio*100}%)")

if len(y_train.shape) == 1 or (len(y_train.shape) == 2 and y_train.shape[-1] == 1):
    num_classes = len(np.unique(y_train))
    print(f"[TRIAL] Sparse labels detected. Classes: {num_classes}")
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
else:
    num_classes = y_train.shape[-1]
    print(f"[TRIAL] Categorical labels detected. Classes: {num_classes}")

print(f"[TRIAL] Dataset classes: {num_classes}")

model = keras.models.load_model('/tmp/customized_model.h5')

if model.output_shape[-1] != num_classes:
    print(f"[TRIAL] ⚠️ Class Mismatch: Model={model.output_shape[-1]}, Data={num_classes}")
    print("[TRIAL] 🔧 Replacing final layer...")
    x = model.layers[-2].output
    output = keras.layers.Dense(num_classes, activation='softmax', name='adaptive_output')(x)
    model = keras.Model(inputs=model.input, outputs=output)
    print(f"[TRIAL] ✓ New output shape: {model.output_shape}")

if freeze_backbone:
    print("[TRIAL] 🧊 Freezing backbone (all layers except last 5)...")
    for layer in model.layers[:-5]:
        layer.trainable = False
else:
    print("[TRIAL] 🔥 Unfrozen backbone: Model will learn on all layers.")

if optimizer_name.lower() == 'sgd':
    opt = keras.optimizers.SGD(learning_rate=lr)
else:
    opt = keras.optimizers.Adam(learning_rate=lr)

expected_shape = model.input_shape[1:3]
print(f"Model expects: {expected_shape}, Data has: {x_train.shape[1:3]}")

def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = x / 255.0
    x = tf.image.resize(x, expected_shape)
    return x, y

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

def augment(x, y):
    x = data_augmentation(x)
    return x, y

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(1000).map(preprocess).batch(batch_size).map(augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(f"[TRIAL] 🚀 Starting training (First epoch may be slow due to resizing/caching)...")
history = model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)

if not IS_RETRAIN:
    val_accuracy = history.history['val_accuracy'][-1]
    print(f"[TRIAL] Validation accuracy: {val_accuracy:.4f}")
    nni.report_final_result(val_accuracy)
else:
    model.save('best_model.h5')
    print(f"[TRIAL] ✅ Best model saved.")
