import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from keras_tuner import RandomSearch
from keras_tuner import HyperParameters

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

# Load back X_train, X_test, y_train, and y_test
X_train = np.loadtxt('train_data/X_train.txt')
X_test = np.loadtxt('train_data/X_test.txt')
y_train = np.loadtxt('train_data/y_train.txt')
y_test = np.loadtxt('train_data/y_test.txt')

# Reshape the data to 3D for CNN input (samples, time steps, features)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_train_reshaped = y_train.reshape((-1, 1))
y_test_reshaped = y_test.reshape((-1, 1))

# Define the CNN-RNN hybrid model
def build_model(hp):
    model = keras.Sequential()
    
    # CNN layers for feature extraction
    model.add(keras.layers.Conv1D(filters=hp.Int('conv_filters', min_value=16, max_value=64, step=16),
                                   kernel_size=hp.Int('conv_kernel', min_value=3, max_value=5, step=1),
                                   activation='relu',
                                   input_shape=(X_train_reshaped.shape[1], 1)))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Dropout(rate=hp.Float('dropout_cnn', min_value=0.0, max_value=0.5, step=0.1)))

    # LSTM layers for temporal modeling
    model.add(keras.layers.LSTM(units=hp.Int('lstm_units', min_value=16, max_value=64, step=16),
                                return_sequences=True))
    model.add(keras.layers.Dropout(rate=hp.Float('dropout_rnn', min_value=0.0, max_value=0.5, step=0.1)))

    # Output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='keras_tuning',
    project_name='cnn_rnn_tuner')

# Perform hyperparameter search
tuner.search(X_train_reshaped, y_train_reshaped,
                epochs=10,
                batch_size=16,
                validation_data=(X_test_reshaped, y_test_reshaped))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
# Save the Keras model to an HDF5 file
best_model.save("hybrid_cnn_rnn.h5")

# Get best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Evaluate the best model
evaluation = best_model.evaluate(X_test_reshaped, y_test_reshaped)
# Print evaluation results
print(evaluation)

# # Load the saved model
# model = load_model("hybrid_cnn_rnn.h5")

# # Convert the Keras model to TensorFlow Lite format
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# # Save the TensorFlow Lite model to a file
# with open("hybrid_cnn_rnn.tflite", "wb") as f:
#     f.write(tflite_model)