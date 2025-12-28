
import nni
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define Retrain Mode
IS_RETRAIN = os.getenv('IS_RETRAIN', '0') == '1'

# Check Retrain Mode
if IS_RETRAIN:
    print("[TRIAL] Retraining mode enabled")
else:
    print("[TRIAL] NNI mode enabled")

# Load NNI or environment parameters
if not IS_RETRAIN:
    params = nni.get_next_parameter()
else:
    params = json.loads(os.getenv('BEST_PARAMS', '{}'))

# Hyperparameters
learning_rate = params.get('learning_rate', 0.001)
batch_size = params.get('batch_size', 32)
optimizer_name = params.get('optimizer', 'Adam')
freeze_mode = params.get('freeze_mode', 'freeze_base')

# Load dataset
dataset_path = '/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10'
x_train = np.load(f'{dataset_path}/x_train.npy')
y_train = np.load(f'{dataset_path}/y_train.npy')
x_test = np.load(f'{dataset_path}/x_test.npy')
y_test = np.load(f'{dataset_path}/y_test.npy')

# Ensure one-hot encoding
if y_train.ndim == 1:
    y_train = np.eye(np.max(y_train)+1)[y_train]
if y_test.ndim == 1:
    y_test = np.eye(np.max(y_test)+1)[y_test]

# Load model
model_path = '/home/mrusso/.stm32_ai_models/mobilenetv2_224.h5'
model = keras.models.load_model(model_path)

# Replace final layer if mismatch
if y_train.shape[1] != model.output_shape[-1]:
    new_output = keras.layers.Dense(y_train.shape[1], activation='softmax')(model.layers[-2].output)
    model = keras.Model(inputs=model.input, outputs=new_output)

# Freeze logic
if freeze_mode == 'freeze_base':
    for layer in model.layers[:-2]:
        layer.trainable = False
else:
    for layer in model.layers:
        layer.trainable = True

# Optimizer
if optimizer_name == 'Adam':
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
else:
    opt = keras.optimizers.SGD(learning_rate=learning_rate)

# Resize images to match model input
expected_shape = model.input_shape[1:3]
def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, expected_shape)
    return image, label

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
    model.save('best_model.h5')
    print(f"[TRIAL] ✅ Model saved to {os.path.abspath('best_model.h5')}")
