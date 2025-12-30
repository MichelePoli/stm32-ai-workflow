
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
    new_output = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=model.input, outputs=new_output)

    print("[TRIAL] ✅ Final layer replaced.")

# Compile
# --- Ensure model architecture matches new final layer if replaced
# No need to recompile here; we will recompile after adjustments

# --- Compile after potential layer replacement ---
# Compile
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(
    keras.utils.image_dataset_from_directory(
        '/mnt/shared-storage/mrusso/STM32CubeMX/data/real_datasets/cifar10',
        label_mode='categorical',
        validation_split=0.2,
        subset='training',
        seed=42,
        image_size=(224, 224),
        batch_size=batch_size
    ),
    epochs=3,
    verbose=0
)

# Report (Only in NNI mode)
if not IS_RETRAIN:
    val_accuracy = history.history['val_accuracy'][-1]
    nni.report_final_result(val_accuracy)
else:
    # Save Model (Only in Retrain mode)
    output_path = 'best_model.h5'
    model.save(output_path)
    print(f"[TRIAL] ✅ Model saved to {os.path.abspath(output_path)}")
