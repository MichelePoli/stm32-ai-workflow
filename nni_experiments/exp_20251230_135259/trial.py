
import nni
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

IS_RETRAIN = os.environ.get('RETRAIN_MODE', 'false').lower() == 'true'

if IS_RETRAIN:
    print("[TRIAL] 🔄 Retraining mode activated!")
    params = json.loads(os.environ.get('RETRAIN_PARAMS', '{}'))
else:
    params = nni.get_next_parameter()

learning_rate = params.get('learning_rate', 0.001)
batch_size = params.get('batch_size', 32)
optimizer_name = params.get('optimizer', 'Adam')
freeze_mode = params.get('freeze_mode', 'none')

print(f"[TRIAL] Params: lr={learning_rate}, bs={batch_size}, opt={optimizer_name}, freeze={freeze_mode}")

# Path to the base pretrained model
base_model_path = '/home/mrusso/.stm32_...'

# Load the base model
base_model = keras.models.load_model(base_model_path)

# Freeze layers based on freeze_mode
if freeze_mode == 'full':
    for layer in base_model.layers:
        layer.trainable = False
elif freeze_mode == 'partial':
    for layer in base_model.layers[:-10]:
        layer.trainable = False
# 'none' means all layers remain trainable

# Replace the final dense layer to match CIFAR‑10 classes
num_classes = 10
if base_model.output_shape[-1] != num_classes:
    # Use the output of the second last layer as input to the new dense layer
    x = base_model.layers[-2].output
    new_output = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=base_model.input, outputs=new_output)
else:
    model = base_model

# Compile the model with the selected optimizer
if optimizer_name == 'SGD':
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
else:
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load CIFAR‑10 data from the provided directory
data_dir = '/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10'
x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

# Preprocessing function
target_height, target_width = model.input_shape[1], model.input_shape[2]
def preprocess(img, label):
    img = tf.image.resize(img, [target_height, target_width])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)

if not IS_RETRAIN:
    val_acc = history.history['val_accuracy'][-1]
    print(f"[TRIAL] Validation accuracy: {val_acc:.4f}")
    nni.report_final_result(val_acc)
else:
    model.save('best_model.h5')
    print("[TRIAL] ✅ Model saved to best_model.h5")
