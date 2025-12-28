
import nni
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

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
freeze_mode = params.get('freeze_mode', 'freeze_base')

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Load model
model_path = '/home/mrusso/.stm32_ai_models/mobilenetv2_224.h5'
model = keras.models.load_model(model_path)

# --- PARAMETER APPLICATION ---

# 1. Freeze Logic
if freeze_mode == 'freeze_base':
    print("Freezing base layers...")
    # Freeze all except last 5 layers
    for layer in model.layers[:-5]:
        layer.trainable = False
else:
    print("Unfreezing all layers...")
    for layer in model.layers:
        layer.trainable = True

# 2. Optimizer Selection
if optimizer_name == 'SGD':
    opt = keras.optimizers.SGD(learning_rate=lr)
else:
    opt = keras.optimizers.Adam(learning_rate=lr)

# --- SHAPE FIX: Resize images to model input size ---
expected_shape = model.input_shape[1:3]  # (H, W)
print(f"Model expects: {expected_shape}, Data has: {x_train.shape[1:3]}")

def preprocess(x, y):
    # Resize to expected shape
    x = tf.image.resize(x, expected_shape)
    return x, y

# Create tf.data pipelines
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Compile
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=3,
                    verbose=0)

# Report (Only in NNI mode)
if not IS_RETRAIN:
    val_accuracy = history.history['val_accuracy'][-1]
    nni.report_final_result(val_accuracy)
else:
    # Save Model (Only in Retrain mode)
    output_path = 'best_model.h5'
    model.save(output_path)
    print(f"[TRIAL] ✅ Model saved to {os.path.abspath(output_path)}")
