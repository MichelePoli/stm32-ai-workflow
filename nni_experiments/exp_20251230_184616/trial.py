
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
    print(f"[TRIAL] Labels already categorical with {num_classes} classes.")

# Load model
model = keras.models.load_model('/home/mrusso/.stm32_ai_models/mobilenetv2_224.h5')

# Replace final layer if classes mismatch
if num_classes != model.output_shape[-1]:
    print("[TRIAL] Replacing final layer to match dataset classes.")
    # Assume last two layers are Dense and softmax
    last_hidden = model.layers[-2].output
    new_output = keras.layers.Dense(num_classes, activation='softmax')(last_hidden)
    model = keras.models.Model(inputs=model.input, outputs=new_output)

# Resize images to model input size
def preprocess(example):
    img, label = example
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    return img, label

# --- INPUT SIZE ADJUSTMENT: Resize to 224x224 ---
expected_shape = model.input_shape[1:3] # (224, 224)

# Prepare datasets
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.map(preprocess).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Compile
if optimizer_name == 'Adam':
    opt = keras.optimizers.Adam(learning_rate=lr)
elif optimizer_name == 'SGD':
    opt = keras.optimizers.SGD(learning_rate=lr)
else:
    opt = keras.optimizers.Adam(learning_rate=lr)  # Default fallback

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(train_ds, 
                    validation_data=val_ds,
                    epochs=3, verbose=0)

# Report (Only in NNI mode)
if not IS_RETRAIN:
    val_accuracy = history.history['val_accuracy'][-1]
    nni.report_final_result(val_accuracy)
else:
    # Save Model (Only in Retrain mode)
    output_path = 'best_model.h5'
    model.save(output_path)
    print(f"[TRIAL] ✅ Model saved to {os.path.abspath(output_path)}")
