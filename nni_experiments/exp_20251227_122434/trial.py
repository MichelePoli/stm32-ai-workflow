
import nni
import numpy as np
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

# Load model
model = keras.models.load_model('/home/mrusso/.stm32_ai_models/mobilenetv2_224.h5')

# Compile
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, 
                    validation_data=(x_test, y_test),
                    epochs=5, batch_size=batch_size, verbose=0)

# Report
val_accuracy = history.history['val_accuracy'][-1]
nni.report_final_result(val_accuracy)
