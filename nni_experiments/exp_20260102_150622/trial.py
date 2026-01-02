
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

# Load data
x_train = np.load(f'/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/x_train.npy')
y_train = np.load(f'/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/y_train.npy')
x_test = np.load(f'/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/x_test.npy')
y_test = np.load(f'/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/y_test.npy')

# --- DATA PREP --- #
# Convert to categorical if needed
if y_train.ndim == 1:
    num_classes = y_train.shape[0]  # Assuming labels are class indices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
else:
    num_classes = y_train.shape[-1]

print(f"[TRIAL] Dataset has {x_train.shape[0]} training samples and {x_test.shape[0]} test samples.")
print(f"[TRIAL] Number of classes: {num_classes}")

# Load the MobileNetV2 model
model = keras.models.load_model('/home/mrusso/.stm32_ai_models/mobilenetv2_224.h5')

# --- OUTPUT ADAPTATION ---
if model.output_shape[-1] != num_classes:
    print("[TRIAL] ⚠️ Adapting final layer to match number of classes.")
    # Remove the last dense layer
    if isinstance(model.layers[-1], keras.layers.Dense):
        x = model.layers[-2].output
        output = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = keras.Model(inputs=model.input, outputs=output)
    else:
        # For other architectures, rebuild from scratch
        inputs = keras.Input(shape=model.input_shape[1:])
        x = inputs
        for layer in model.layers:
            if isinstance(layer, keras.layers.Dense) and layer.output_shape[-1] != num_classes:
                continue
            x = layer(x)
        model = keras.Model(inputs=inputs, outputs=keras.layers.Dense(num_classes, activation='softmax')(x))
    print(f"[TRIAL] New output shape: {model.output_shape}")

# --- OPTIMIZER ---
if optimizer_name.lower() == 'sgd':
    opt = keras.optimizers.SGD(learning_rate=lr)
else:
    opt = keras.optimizers.Adam(learning_rate=lr)

# --- DATA PREPROCESSING ---
expected_input_shape = model.input_shape[1:3]  # (height, width)

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, expected_input_shape)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- COMPILATION ---
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- TRAINING ---
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=3,
                    verbose=1)

# --- REPORT OR SAVE ---
if not IS_RETRAIN:
    val_accuracy = history.history['val_accuracy'][-1]
    nni.report_final_result(val_accuracy)
else:
    model.save('best_model.h5')
    print("[TRIAL] ✅ Best model saved.")
