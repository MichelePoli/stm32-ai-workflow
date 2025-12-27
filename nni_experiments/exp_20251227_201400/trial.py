
import nni
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Get hyperparameters
params = nni.get_next_parameter()
lr = params.get('learning_rate', 0.001)
batch_size = params.get('batch_size', 32)

# Load data
x_train = np.load('/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/x_train.npy')
y_train = np.load('/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/y_train.npy')
x_test = np.load('/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/x_test.npy')
y_test = np.load('/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10/y_test.npy')

# Convert labels to one-hot
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Load base model
base_model = keras.models.load_model('/home/mrusso/.stm32_ai_models/mobilenetv2_224.h5')

# Replace the final layer to match CIFAR-10 classes
x = base_model.input
# Exclude the last layer (assumed to be the original 1000-class Dense)
# We'll take the output of the penultimate layer
penultimate = base_model.layers[-2].output
new_output = keras.layers.Dense(num_classes, activation='softmax')(penultimate)
model = keras.Model(inputs=x, outputs=new_output)

# Freeze all layers except the new final layer
for layer in model.layers[:-1]:
    layer.trainable = False

# --- SHAPE FIX: Resize images if needed ---
expected_shape = (224, 224)  # (H, W)
print(f"Model expects: {expected_shape}, Data has: {x_train.shape[1:3]}")

def preprocess(x, y):
    # Resize and normalize
    x = tf.image.resize(x, expected_shape)
    x = x / 255.0
    return x, y

# Create efficient tf.data pipeline
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Compile
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=3,
                    verbose=0)

# Report
val_accuracy = history.history['val_accuracy'][-1]
nni.report_final_result(val_accuracy)
