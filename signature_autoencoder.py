import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib

# Preprocessing function
def preprocess_data(file):
    data = pd.read_csv(file)
    selected_data = data.drop(columns=['timestamp'])
    normalized_data = pd.DataFrame(selected_data, columns=selected_data.columns)
    return normalized_data

folder_name = 'extract_feature'
filename = [
    # "clearwrite_captured_SWEEP_REC_2024-07-07 15h57m07s_raspi4_b_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 16h31m15s_raspi4_b_on.csv",
    # "clearwrite_captured_SWEEP_REC_2024-07-07 18h42m08s_raspi4_a_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 19h17m56s_raspi4_a_on.csv",
    # "clearwrite_captured_SWEEP_REC_2024-07-07 14h44m03s_esp32_on.csv",
]

# Load and preprocess data
data = []
for f in filename:
    print(f"\nProcessing file: {f}")
    file_data = preprocess_data(os.path.join(folder_name, f))
    data.append(file_data)

X = np.concatenate(data)

# Train-test split
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Define the autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 128  # Adjust based on desired compression

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

# Save the autoencoder and encoder models
autoencoder.save('signature/model/autoencoder_model.h5')
encoder.save('signature/model/encoder_model.h5')

known_signatures = {}
for i, f in enumerate(filename):
    X_known = preprocess_data(os.path.join(folder_name, f))
    known_signatures[f"camera_{i}"] = encoder.predict(X_known).mean(axis=0)

# Save known signatures
joblib.dump(known_signatures, 'signature/known_signatures/known_signatures.pkl')