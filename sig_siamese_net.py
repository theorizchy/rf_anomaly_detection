import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

# cam_1_sig = np.loadtxt('sig_enc/cam_1_sig')
# cam_2_sig = np.loadtxt('sig_enc/cam_2_sig')
# cam_3_sig = np.loadtxt('sig_enc/cam_3_sig')
# cam_4_sig = np.loadtxt('sig_enc/cam_4_sig')
# cam_5_sig = np.loadtxt('sig_enc/cam_5_sig')
# cam_6_sig = np.loadtxt('sig_enc/cam_6_sig')
# cam_7_sig = np.loadtxt('sig_enc/cam_7_sig')

cam_1_sig = np.loadtxt('sig_enc/cam_1_signature')
cam_2_sig = np.loadtxt('sig_enc/cam_2_signature')
cam_3_sig = np.loadtxt('sig_enc/cam_3_signature')
cam_4_sig = np.loadtxt('sig_enc/cam_4_signature')
cam_5_sig = np.loadtxt('sig_enc/cam_5_signature')

def create_triplets(anchor_signatures, positive_signatures, negative_signatures):
    num_samples = anchor_signatures.shape[0]
    anchor_indices = np.random.choice(num_samples, size=num_samples, replace=True)
    positive_indices = np.random.choice(num_samples, size=num_samples, replace=True)
    negative_indices = np.random.choice(num_samples, size=num_samples, replace=True)
    return anchor_signatures[anchor_indices], positive_signatures[positive_indices], negative_signatures[negative_indices]

cam_1_train, cam_1_val = train_test_split(cam_1_sig, test_size=0.2, random_state=42)
cam_2_train, cam_2_val = train_test_split(cam_2_sig, test_size=0.2, random_state=42)
cam_3_train, cam_3_val = train_test_split(cam_3_sig, test_size=0.2, random_state=42)
cam_4_train, cam_4_val = train_test_split(cam_4_sig, test_size=0.2, random_state=42)
cam_5_train, cam_5_val = train_test_split(cam_5_sig, test_size=0.2, random_state=42)
# cam_6_train, cam_6_val = train_test_split(cam_6_sig, test_size=0.2, random_state=42)
# cam_7_train, cam_7_val = train_test_split(cam_7_sig, test_size=0.2, random_state=42)

anchor_train, positive_train, negative_train = create_triplets(cam_1_train, cam_2_train, cam_3_train)
anchor_val, positive_val, negative_val = create_triplets(cam_1_val, cam_2_val, cam_3_val)

input_shape = (128,)
anchor_input = Input(shape=input_shape)
positive_input = Input(shape=input_shape)
negative_input = Input(shape=input_shape)

embedding_layer = Dense(128, activation='relu')
anchor_embedding = embedding_layer(anchor_input)
positive_embedding = embedding_layer(positive_input)
negative_embedding = embedding_layer(negative_input)

def triplet_loss(_, y_pred):
    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    margin = 0.2
    return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))

output = Lambda(lambda x: tf.concat(x, axis=1))([anchor_embedding, positive_embedding, negative_embedding])
siamese_net = Model(inputs=[anchor_input, positive_input, negative_input], outputs=output)

siamese_net.compile(optimizer='adam', loss=triplet_loss)

siamese_net.fit([anchor_train, positive_train, negative_train], np.zeros((anchor_train.shape[0], 3)),
                epochs=50, batch_size=32, validation_split=0.2)

# Save the trained Siamese network
siamese_net.save('sig_enc/model/siamese_net_model.h5')

# Evaluate on validation set or test set
y_val_pred = siamese_net.predict([anchor_val, positive_val, negative_val])

# Example of evaluating embeddings (you may need to adjust this based on your actual evaluation criteria)
from sklearn.metrics import pairwise_distances

# Calculate distances between anchor and positive embeddings
pos_distances = pairwise_distances(anchor_val, positive_val, metric='euclidean')
# Calculate distances between anchor and negative embeddings
neg_distances = pairwise_distances(anchor_val, negative_val, metric='euclidean')

# Example of evaluating based on distances
pos_mean_dist = np.mean(pos_distances)
neg_mean_dist = np.mean(neg_distances)

print(f"Mean positive distance: {pos_mean_dist}")
print(f"Mean negative distance: {neg_mean_dist}")
