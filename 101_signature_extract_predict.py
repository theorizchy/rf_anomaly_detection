import os
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
import random
from sklearn.utils import check_random_state

# Set a global seed
SEED = 777
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random_state = check_random_state(SEED)

known_signatures = {}
all_features = []
len_files = []

NUM_OF_PEAKS = 5
RETRAIN_ENCODER = False
folder_name = 'extract_peaks'

if RETRAIN_ENCODER:
    scaler = StandardScaler()
else:
    scaler = joblib.load('signature/model/sig_scaler.pkl')

def extract_feature(file, n=NUM_OF_PEAKS):
    data = pd.read_csv(file)
    if n == 5:
        # Calculate the ratios of the powers
        data['ratio_3_to_1'] = data['peak_3_pwr'] / data['peak_1_pwr']
        data['ratio_3_to_2'] = data['peak_3_pwr'] / data['peak_2_pwr']
        data['ratio_3_to_4'] = data['peak_3_pwr'] / data['peak_4_pwr']
        data['ratio_3_to_5'] = data['peak_3_pwr'] / data['peak_5_pwr']

        # Relative power strength from total
        sum_peak_powers = data[['peak_1_pwr', 'peak_2_pwr', 'peak_3_pwr', 'peak_4_pwr', 'peak_5_pwr']].sum(axis=1)
        # Normalize the peak powers
        data['norm_peak_1_pwr'] = data['peak_1_pwr'] / sum_peak_powers
        data['norm_peak_2_pwr'] = data['peak_2_pwr'] / sum_peak_powers
        data['norm_peak_3_pwr'] = data['peak_3_pwr'] / sum_peak_powers
        data['norm_peak_4_pwr'] = data['peak_4_pwr'] / sum_peak_powers
        data['norm_peak_5_pwr'] = data['peak_5_pwr'] / sum_peak_powers

        # Calculate the widths between the frequencies
        span_freq = data['peak_5_freq'] - data['peak_1_freq']
        width_1_to_2 = data['peak_2_freq'] - data['peak_1_freq']
        width_2_to_3 = data['peak_3_freq'] - data['peak_2_freq']
        width_3_to_4 = data['peak_4_freq'] - data['peak_3_freq']
        width_4_to_5 = data['peak_5_freq'] - data['peak_4_freq']

        # Normalize the widths by the overall span of the frequencies
        data['rel_width_1_to_2'] = width_1_to_2 / span_freq
        data['rel_width_2_to_3'] = width_2_to_3 / span_freq
        data['rel_width_3_to_4'] = width_3_to_4 / span_freq
        data['rel_width_4_to_5'] = width_4_to_5 / span_freq

        features = ['timestamp',
                    'ratio_3_to_1', 'ratio_3_to_2', 'ratio_3_to_4', 'ratio_3_to_5',
                    'peak_1_freq', 'peak_2_freq', 'peak_3_freq', 'peak_4_freq', 'peak_5_freq',
                    'norm_peak_1_pwr', 'norm_peak_2_pwr', 'norm_peak_3_pwr', 'norm_peak_4_pwr', 'norm_peak_5_pwr',
                    'rel_width_1_to_2', 'rel_width_2_to_3', 'rel_width_3_to_4', 'rel_width_4_to_5',
                    'var_interval_1', 'var_interval_2', 'var_interval_3', 'var_interval_4', 'var_interval_5',
                    'skewness_interval_1', 'skewness_interval_2', 'skewness_interval_3', 'skewness_interval_4', 'skewness_interval_5',
                    'kurtosis_interval_1', 'kurtosis_interval_2', 'kurtosis_interval_3', 'kurtosis_interval_4', 'kurtosis_interval_5'
                    ]
    elif n == 3:
        # Calculate the ratios of the powers
        data['ratio_2_to_1'] = data['peak_2_pwr'] / data['peak_1_pwr']
        data['ratio_2_to_3'] = data['peak_2_pwr'] / data['peak_3_pwr']

        # Relative power strength from total
        sum_peak_powers = data[['peak_1_pwr', 'peak_2_pwr', 'peak_3_pwr']].sum(axis=1)
        # Normalize the peak powers
        data['norm_peak_1_pwr'] = data['peak_1_pwr'] / sum_peak_powers
        data['norm_peak_2_pwr'] = data['peak_2_pwr'] / sum_peak_powers
        data['norm_peak_3_pwr'] = data['peak_3_pwr'] / sum_peak_powers

        # Calculate the widths between the frequencies
        span_freq = data['peak_3_freq'] - data['peak_1_freq']
        width_1_to_2 = data['peak_2_freq'] - data['peak_1_freq']
        width_2_to_3 = data['peak_3_freq'] - data['peak_2_freq']

        # Normalize the widths by the overall span of the frequencies
        data['rel_width_1_to_2'] = width_1_to_2 / span_freq
        data['rel_width_2_to_3'] = width_2_to_3 / span_freq

        features = ['timestamp',
                    'ratio_2_to_1', 'ratio_2_to_3',
                    'peak_1_freq', 'peak_2_freq', 'peak_3_freq',
                    'norm_peak_1_pwr', 'norm_peak_2_pwr', 'norm_peak_3_pwr',
                    'rel_width_1_to_2', 'rel_width_2_to_3',
                    'var_interval_1', 'var_interval_2', 'var_interval_3',
                    'skewness_interval_1', 'skewness_interval_2', 'skewness_interval_3',
                    'kurtosis_interval_1', 'kurtosis_interval_2', 'kurtosis_interval_3'
                    ]

    # Save to CSV
    output_folder = 'extract_feature'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, os.path.basename(file))

    # Select only the newly created features
    new_features = data[features]
    # Save the new features to a new CSV file
    new_features.to_csv(output_file, index=False)

    print(f"Extracted features saved to: {output_file}")

    # Return the feature data for further processing
    return new_features

def preprocess_data(data, train=True):
    global scaler
    if train:
        scaled_data = scaler.fit_transform(data.drop(columns=['timestamp']))
    else:
        scaled_data = scaler.transform(data.drop(columns=['timestamp']))
    return scaled_data

def find_most_similar_signature(new_signature, known_signatures):
    similarities = {camera: cosine_similarity([new_signature], sig)[0][0] for camera, sig in known_signatures.items()}
    most_similar_camera = max(similarities, key=similarities.get)
    return most_similar_camera, similarities[most_similar_camera]

def split_encoded_data(data, lengths):
    global known_signatures, encoder
    assert len(data) == sum(lengths), "The sum of lengths must be equal to the length of encoded_data"
    current_position = 0

    for i, length in enumerate(lengths):
        key = f"camera_{i}"
        chunked_data = data[current_position:current_position + length]
        avg_chunked_data = np.mean(chunked_data, axis=0).reshape(1, -1) 
        known_signatures[key] = encoder.predict(avg_chunked_data).reshape(1, -1)
        current_position += length

    return known_signatures

def create_autoencoder(input_dim, encoding_dim=128):
    input_layer = Input(shape=(input_dim,))
    x = Dense(128, activation="relu")(input_layer)
    x = BatchNormalization()(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    encoded = Dense(encoding_dim, activation="relu")(x)
    
    x = Dense(64, activation="relu")(encoded)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    decoded = Dense(input_dim, activation="sigmoid")(x)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    return autoencoder, encoder

def is_known_signature(new_sample, threshold):
    new_sample_encoded = encoder.predict(new_sample.reshape(1, -1))
    new_sample_reconstructed = autoencoder.predict(new_sample.reshape(1, -1))
    new_sample_error = np.mean(np.square(new_sample - new_sample_reconstructed))
    return new_sample_error < threshold

if RETRAIN_ENCODER:
    filename = [
        "clearwrite_captured_SWEEP_REC_2024-07-07 15h57m07s_raspi4_b_on.csv",
        "clearwrite_captured_SWEEP_REC_2024-07-07 18h42m08s_raspi4_a_on.csv",
        "clearwrite_captured_SWEEP_REC_2024-07-13 12h20m09s_esp32_on.csv",
    ]

    # Process known signatures
    # Loop through the filenames, extract features and append to the list
    for idx, f in enumerate(filename):
        data = extract_feature(os.path.join(folder_name, f))
        all_features.append(data)
        len_files.append(data.shape[0])

    # Combine features into a single DataFrame
    all_features_df = pd.concat(all_features, ignore_index=True)

    # Preprocess data
    preprocessed_data = preprocess_data(all_features_df, train=True)

    # Split data into train and test sets
    X_train, X_test = train_test_split(preprocessed_data, test_size=0.2, random_state=random_state)

    # Create and compile the autoencoder
    input_dim = X_train.shape[1]
    autoencoder, encoder = create_autoencoder(input_dim, encoding_dim=input_dim) # Change encoding_dim to input_dim
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    history = autoencoder.fit(X_train, X_train, 
                            epochs=50, batch_size=64,
                            use_multiprocessing=True, 
                            validation_data=(X_test, X_test))
    encoded_data = encoder.predict(preprocessed_data)

    # Calculate reconstruction errors on the training set
    reconstructed_data = autoencoder.predict(preprocessed_data)
    reconstruction_errors = np.mean(np.square(preprocessed_data - reconstructed_data), axis=1)

    threshold = np.mean(reconstruction_errors) + 3 * np.std(reconstruction_errors)
    # Generate known signatures in low-latent format
    known_signatures = split_encoded_data(preprocessed_data, lengths=len_files)

    # Save known signatures
    joblib.dump(known_signatures, 'signature/model/known_signatures.pkl')
    joblib.dump(scaler, 'signature/model/sig_scaler.pkl')
    # Save encoder and autoencoder
    autoencoder.save('signature/model/autoencoder.h5')
    encoder.save('signature/model/encoder.h5')
    # Save threshold value
    np.savetxt('signature/model/threshold.txt', [threshold])
else:
    # Load Known_signatures
    known_signatures = joblib.load('signature/model/known_signatures.pkl')
    scaler = joblib.load('signature/model/sig_scaler.pkl')
    # Load the autoencoder model
    autoencoder = load_model('signature/model/autoencoder.h5')
    # Load the encoder model
    encoder = load_model('signature/model/encoder.h5')
    # Load the threshold value
    threshold = np.loadtxt('signature/model/threshold.txt')

#####################################################################################
# Compare new data
#####################################################################################
new_data_files = [
    "clearwrite_captured_SWEEP_REC_2024-07-07 16h31m15s_raspi4_b_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 19h17m56s_raspi4_a_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 14h44m03s_esp32_on.csv",
]

print(f"\n==================== START PREDICTING ====================")

summary = []
for f in new_data_files:
    print(f"\nComparing new data file: {f}")
    extracted_features = extract_feature(os.path.join(folder_name, f))
    scaled_features = preprocess_data(extracted_features, train=False)
    
    predictions = {}
    # Find the most similar known signature
    for sample in scaled_features:
        new_signature = encoder.predict(sample.reshape(1, -1)).flatten()
        most_similar_camera, similarity_score = find_most_similar_signature(new_signature, known_signatures)
        is_known = (similarity_score > 0.5) and is_known_signature(sample, threshold)
        # Define a threshold for similarity score to classify known/unknown
        if is_known:
            predictions[most_similar_camera] = predictions.get(most_similar_camera, 0) + 1
            print(f"[KNOWN] The new camera data is most similar to the {most_similar_camera} camera with Similarity score: {similarity_score}%")
        else:
            predictions['unknown'] = predictions.get('unknown',0) + 1
            print(f"[UNKNOWN] The new camera data does not match any known cameras. Similarity score: {similarity_score}")

    summary.append(predictions)

print('==== SUMMARY ====')
for idx, predict in enumerate(summary):
    print(f'>> For file {idx}:')
    for (key, val) in sorted(predict.items()):
        print(f'   - [{key}]: {val}')

