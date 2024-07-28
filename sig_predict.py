import numpy as np
from keras.models import load_model
import tensorflow as tf
from keras.models import Model

# Define the triplet loss function
def triplet_loss(_, y_pred):
    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    margin = 0.2
    return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))

# Load the trained Siamese network
siamese_net = load_model('sig_enc/model/siamese_net_model.h5', custom_objects={'triplet_loss': triplet_loss})

# Load the encoder part of the Siamese network
encoder_input = siamese_net.input[1]  # Assuming the encoder input is the first input
encoder_output = siamese_net.layers[-4].output  # Assuming the encoder output is the 4th last layer
encoder = Model(encoder_input, encoder_output)

# Define a function to predict unknown camera signature
def predict_unknown_signature(unknown_sig, known_signatures, threshold=0.5):
    # Compute the embedding for the unknown signature
    unknown_embedding = encoder.predict(unknown_sig)

    for (idx,known_sig) in enumerate(known_signatures):
        # Compute the embedding for the known signature
        known_embedding = encoder.predict(known_sig)

        # Calculate distances
        dists = np.linalg.norm(unknown_embedding - known_embedding, axis=1)
        mean_dist = np.mean(dists)

        print(f'Camera {idx+1} mean distance: {mean_dist}')

        # Check if the unknown signature matches the known signature
        if mean_dist < threshold:
            return True, (idx + 1)

    return False, -1

# Load camera signatures samples
# cam_1_sig = np.loadtxt('sig_enc/cam_1_sig')
# cam_2_sig = np.loadtxt('sig_enc/cam_2_sig')
# cam_3_sig = np.loadtxt('sig_enc/cam_3_sig')
# cam_4_sig = np.loadtxt('sig_enc/cam_4_sig')

cam_1_sig = np.loadtxt('sig_enc/cam_1_signature')
cam_2_sig = np.loadtxt('sig_enc/cam_2_signature')
cam_3_sig = np.loadtxt('sig_enc/cam_3_signature')
cam_4_sig = np.loadtxt('sig_enc/cam_4_signature')
cam_5_sig = np.loadtxt('sig_enc/cam_5_signature')

# Assuming list of unknown signature
known_signatures = [cam_1_sig, cam_2_sig, cam_3_sig, cam_4_sig, cam_5_sig]

# Predicting unknown signature
for sig in [cam_5_sig]:
    print('\nStart predicting ...')
    # Predict whether the unknown signature matches any known signatures
    match_found, cam_num = predict_unknown_signature(sig, known_signatures)

    if match_found:
        print(f"Matches a known camera signature CAM: [{cam_num}].")
    else:
        print("Does not match any known camera signatures.")
