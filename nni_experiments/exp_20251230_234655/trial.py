
import nni
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

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
# Load data
x_train = np.load('/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/x_train.npy')
y_train = np.load('/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/y_train.npy')
x_test = np.load('/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/x_test.npy')
y_test = np.load('/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/y_test.npy')

# --- DATASET FIX: Ensure One-Hot Labels ---
if len(y_train.shape) == 1 or y_train.shape[-1] == 1:
    num_classes = len(np.unique(y_train))
    print(f"[TRIAL] Converting sparse labels to categorical (classes={num_classes})...")
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
else:
    num_classes = y_train.shape[-1]

print(f"[TRIAL] Dataset classes: {num_classes}")

# Load model
model = keras.models.load_model('/home/mrusso/.stm32_ai_models/mobilenetv2_224.h5')

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

# Datasets with caching
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.map(preprocess).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

# FINAL COMPILE (Generate ONLY this block)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(train_ds, 
                    validation_data=val_ds,
                    epochs=3, verbose=0)

# Report or Save
if not IS_RETRAIN:
    val_accuracy = history.history['val_accuracy'][-1]
    nni.report_final_result(val_accuracy)
else:
    model.save('best_model.h5')
    print(f"[TRIAL] ✅ Best model saved.")
