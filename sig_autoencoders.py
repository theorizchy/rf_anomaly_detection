import numpy as np
from keras.layers import Input, Dense, Conv1D, Flatten
from keras.models import Model
import os
import pandas as pd

START_FREQ  = 455500000         # 455.5 MHz 
END_FREQ    = 456500000         # 456.5 MHz
FREQ_BIN_SIZE = 76.2834694      # 76.283 Hz

# Set a seed for random operations
np.random.seed(123)

def preprocess_dataframe(df):
    timestamps = df.iloc[:, 0]
    data = df.iloc[:, 1:]
    num_columns = len(data.columns)
    frequency_values = [START_FREQ + i * FREQ_BIN_SIZE for i in range(num_columns)]
    data.index = pd.to_datetime(timestamps)
    data.index.name = 'Timestamp'
    data.columns = frequency_values
    return data

def aggregate_samples(dfa, t=10):
    num_rows_to_keep = (len(dfa) // t) * t
    truncated_dfa = dfa.iloc[:num_rows_to_keep]
    dfa_reset = truncated_dfa.reset_index()
    grouped_data = dfa_reset.iloc[:, 1:].values.reshape(-1, t, dfa.shape[1])
    min_values = grouped_data.min(axis=1)
    min_values_dfa = pd.DataFrame(min_values, columns=dfa.columns)
    latest_timestamps = dfa_reset.groupby(dfa_reset.index // t)['Timestamp'].max()
    min_values_dfa.insert(0, 'Timestamp', latest_timestamps)
    return min_values_dfa

def augment_signature(sig, noise_factor=0.1, attenuation_factor=0.5):
    noise = np.random.randn(*sig.shape) * noise_factor
    attenuated_sig = sig * np.random.uniform(attenuation_factor, 1.0)
    augmented_sig = attenuated_sig + noise
    return augmented_sig

cam_files = [
    "clearwrite_captured_SWEEP_REC_2024-07-02 21h08m48s_raspi5_b_on.csv",     # cam_1_sig
    "clearwrite_captured_SWEEP_REC_2024-07-03 18h56m52s_raspi4_b_on.csv",     # cam_2_sig
    "clearwrite_captured_SWEEP_REC_2024-07-03 19h41m22s_raspi4_a_on.csv",     # cam_3_sig
    "clearwrite_captured_SWEEP_REC_2024-06-11 21h01m22s_cam_on.csv",          # cam_4_sig
    "clearwrite_captured_SWEEP_REC_2024-07-06 18h26m30s_off.csv",             # cam_5_sig
    "clearwrite_captured_SWEEP_REC_2024-06-11 20h09m34s_cam_off.csv",         # cam_6_sig
    "clearwrite_captured_SWEEP_REC_2024-06-16 12h11m17s_cam_off.csv"          # cam_7_sig
    "clearwrite_captured_SWEEP_REC_2024-07-07 15h57m07s_raspi4_b_on.csv",     # cam_1_signature
    "clearwrite_captured_SWEEP_REC_2024-07-07 16h31m15s_raspi4_b_on.csv",     # cam_2_signature
    "clearwrite_captured_SWEEP_REC_2024-07-07 18h42m08s_raspi4_a_on.csv",     # cam_3_signature
    "clearwrite_captured_SWEEP_REC_2024-07-07 19h17m56s_raspi4_a_on.csv",     # cam_4_signature
    "clearwrite_captured_SWEEP_REC_2024-07-07 14h44m03s_esp32_on.csv",        # cam_5_signature
]

for idx, f in enumerate(cam_files):
    file_path = os.path.join('output_file', f)
    read_file = pd.read_csv(file_path, header=None)
    df = preprocess_dataframe(read_file)
    X = aggregate_samples(dfa=df, t=10)
    X_train = X.drop(columns=['Timestamp'])

    # Augment the signatures
    augmented_X_train = augment_signature(X_train.values)

    input_shape = (X_train.shape[1], 1)
    input_layer = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = Flatten()(x)
    encoded = Dense(128, activation='relu')(x)
    decoded = Dense(input_shape[0], activation='sigmoid')(encoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Combine original and augmented data
    combined_X_train = np.concatenate((X_train.values, augmented_X_train), axis=0)
    combined_X_train = np.expand_dims(combined_X_train, axis=2)

    # Train the autoencoder
    autoencoder.fit(combined_X_train, combined_X_train, epochs=50, batch_size=32, validation_split=0.2)

    # Save encoder part of the model
    encoder = Model(input_layer, encoded)
    X_train_features = encoder.predict(combined_X_train)

    signature_enc_folder = 'sig_enc'
    if not os.path.exists(signature_enc_folder):
        os.makedirs(signature_enc_folder)

    np.savetxt(os.path.join(signature_enc_folder, f'cam_{idx+4}_signature'), X_train_features)
