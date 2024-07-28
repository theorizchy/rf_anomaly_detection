import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

# Preprocessing function
def preprocess_data(file):
    data = pd.read_csv(file)
    selected_data = data.drop(columns=['timestamp'])
    normalized_data = pd.DataFrame(selected_data, columns=selected_data.columns)
    return normalized_data

# Load the encoder model and known signatures
encoder = tf.keras.models.load_model('signature/model/encoder_model.h5')
known_signatures = joblib.load('signature/known_signatures/known_signatures.pkl')

def find_most_similar_signature(new_signature, known_signatures):
    similarities = {camera: cosine_similarity([new_signature], [sig])[0][0] for camera, sig in known_signatures.items()}
    print(similarities)
    most_similar_camera = max(similarities, key=similarities.get)
    return most_similar_camera, similarities[most_similar_camera]

# Predicting known or unknown camera for new data
folder_name = 'extract_feature'

filename = [
    "clearwrite_captured_SWEEP_REC_2024-07-07 15h57m07s_raspi4_b_on.csv",
    # "clearwrite_captured_SWEEP_REC_2024-07-07 16h31m15s_raspi4_b_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 18h42m08s_raspi4_a_on.csv",
    # "clearwrite_captured_SWEEP_REC_2024-07-07 19h17m56s_raspi4_a_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 14h44m03s_esp32_on.csv",
]

for f in filename:
    print(f"\nFor file: {f}")
    X_new = preprocess_data(os.path.join(folder_name, f))
    new_signature = encoder.predict(X_new).mean(axis=0)

    most_similar_camera, similarity_score = find_most_similar_signature(new_signature, known_signatures)

    # Adjust threshold based on your needs
    if similarity_score > 0.8:
        print(f"Most similar to {most_similar_camera} with similarity score: {similarity_score}")
    else:
        print("Unknown camera")
