import os
import csv
import pyvisa
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
from helpers.mail import sendEmail
import joblib
import argparse
import sys
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random
from sklearn.utils import check_random_state
from keras.models import load_model

# Set a global seed
SEED = 777
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random_state = check_random_state(SEED)

# Load Known_signatures
known_signatures = joblib.load('signature/model/known_signatures.pkl')
sig_scaler = joblib.load('signature/model/sig_scaler.pkl')
# Load the autoencoder model
autoencoder = load_model('signature/model/autoencoder.h5')
# Load the encoder model
encoder = load_model('signature/model/encoder.h5')
# Load the threshold value
threshold = np.loadtxt('signature/model/threshold.txt')
# To store summary of predictions
on_off_predictions = {}
signature_predictions = {}

# Limit for consecutive camera ON
CAM_ON_LIMIT = 500
# Prediction threshold for camera ON
CAM_THRESHOLD = 0.7
# Numbers of sample to be taken before actual prediction
SAMPLE_COUNT = 100
NUM_OF_PEAKS = 5

class Helper:
    # Get current timestamp with millisecond
    @staticmethod
    def get_current_timestamp():
        return datetime.now().astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S.%f')

    @staticmethod
    def extract_feature(data, n=NUM_OF_PEAKS):
        # Convert list to DataFrame
        columns = [f'peak_{i+1}_freq' for i in range(n)] + [f'peak_{i+1}_pwr' for i in range(n)] + \
                  [f'var_interval_{i+1}' for i in range(n)] + [f'skewness_interval_{i+1}' for i in range(n)] + [f'kurtosis_interval_{i+1}' for i in range(n)]
        data_df = pd.DataFrame([data], columns=columns)
        
        # Calculate the ratios of the powers
        data_df['ratio_3_to_1'] = data_df['peak_3_pwr'] / data_df['peak_1_pwr']
        data_df['ratio_3_to_2'] = data_df['peak_3_pwr'] / data_df['peak_2_pwr']
        data_df['ratio_3_to_4'] = data_df['peak_3_pwr'] / data_df['peak_4_pwr']
        data_df['ratio_3_to_5'] = data_df['peak_3_pwr'] / data_df['peak_5_pwr']

        # Relative power strength from total
        sum_peak_powers = data_df[['peak_1_pwr', 'peak_2_pwr', 'peak_3_pwr', 'peak_4_pwr', 'peak_5_pwr']].sum(axis=1)
        # Normalize the peak powers
        data_df['norm_peak_1_pwr'] = data_df['peak_1_pwr'] / sum_peak_powers
        data_df['norm_peak_2_pwr'] = data_df['peak_2_pwr'] / sum_peak_powers
        data_df['norm_peak_3_pwr'] = data_df['peak_3_pwr'] / sum_peak_powers
        data_df['norm_peak_4_pwr'] = data_df['peak_4_pwr'] / sum_peak_powers
        data_df['norm_peak_5_pwr'] = data_df['peak_5_pwr'] / sum_peak_powers

        # Calculate the widths between the frequencies
        span_freq = data_df['peak_5_freq'] - data_df['peak_1_freq']
        width_1_to_2 = data_df['peak_2_freq'] - data_df['peak_1_freq']
        width_2_to_3 = data_df['peak_3_freq'] - data_df['peak_2_freq']
        width_3_to_4 = data_df['peak_4_freq'] - data_df['peak_3_freq']
        width_4_to_5 = data_df['peak_5_freq'] - data_df['peak_4_freq']

        # Normalize the widths by the overall span of the frequencies
        data_df['rel_width_1_to_2'] = width_1_to_2 / span_freq
        data_df['rel_width_2_to_3'] = width_2_to_3 / span_freq
        data_df['rel_width_3_to_4'] = width_3_to_4 / span_freq
        data_df['rel_width_4_to_5'] = width_4_to_5 / span_freq

        features = ['ratio_3_to_1', 'ratio_3_to_2', 'ratio_3_to_4', 'ratio_3_to_5',
                    'peak_1_freq', 'peak_2_freq', 'peak_3_freq', 'peak_4_freq', 'peak_5_freq',
                    'norm_peak_1_pwr', 'norm_peak_2_pwr', 'norm_peak_3_pwr', 'norm_peak_4_pwr', 'norm_peak_5_pwr',
                    'rel_width_1_to_2', 'rel_width_2_to_3', 'rel_width_3_to_4', 'rel_width_4_to_5',
                    'var_interval_1', 'var_interval_2', 'var_interval_3', 'var_interval_4', 'var_interval_5',
                    'skewness_interval_1', 'skewness_interval_2', 'skewness_interval_3', 'skewness_interval_4', 'skewness_interval_5',
                    'kurtosis_interval_1', 'kurtosis_interval_2', 'kurtosis_interval_3', 'kurtosis_interval_4', 'kurtosis_interval_5'
                    ]
        return data_df[features]

    @staticmethod
    def find_peak_values(data, start_freq=455500000, stop_freq=456500000, center_freq=456000000, delta=50000, top_n=5):
        # Calculate the interval
        num_points = len(data)
        interval = (stop_freq - start_freq) / num_points
        output_data = []

        # Function to find the top N peaks in the given measurements
        def find_top_n_peaks(measurements, top_n):
            # Calculate the index range for the target frequencies
            center_index = int((center_freq - start_freq) / interval)
            delta_index = int(delta / interval)
            
            # search range for central frequency
            search_ranges = [range(center_index - delta_index, center_index + delta_index - 1)]
            # Populate search range for peak around central freq
            for i in range(1, 1+(top_n-1)//2):
                span = 2 * delta_index
                left_edge = center_index - (delta_index * (2 * i + 1)) 
                right_edge = left_edge + span - 1
                search_ranges.append(range(left_edge, right_edge))
                right_edge = center_index + (delta_index * (2 * i + 1)) - 1
                left_edge = right_edge - span
                search_ranges.append(range(left_edge, right_edge))

            all_selected_freqs = []
            all_selected_powers = []
            all_stats = []
            
            for search_indices in search_ranges:
                search_range = measurements[search_indices]
                peaks, properties = find_peaks(search_range, height=-120)
                peak_indices = np.array([search_indices.start + p for p in peaks])
                peak_freqs = np.array([start_freq + p * interval for p in peak_indices])
                peak_powers = properties['peak_heights']
                
                all_selected_freqs.extend(peak_freqs)
                all_selected_powers.extend(peak_powers)

                # Calculate statistics for this interval
                if len(search_range) > 0:
                    var = np.var(search_range)
                    skewness = skew(search_range)
                    kurt = kurtosis(search_range)
                else:
                    var = skewness = kurt = np.nan

                all_stats.append((var, skewness, kurt))

            if len(all_selected_powers) < top_n:
                top_n = len(all_selected_powers)
            
            top_peaks = np.argsort(all_selected_powers)[-top_n:]

            selected_freqs = np.array(all_selected_freqs)[top_peaks]
            selected_powers = np.array(all_selected_powers)[top_peaks]
            selected_stats = np.array(all_stats)

            # Sort the selected peaks by frequency
            sorted_indices = np.argsort(selected_freqs)
            selected_freqs = selected_freqs[sorted_indices]
            selected_powers = selected_powers[sorted_indices]
            selected_stats = selected_stats[sorted_indices]

            return selected_freqs, selected_powers, selected_stats

        # Convert values to float and extract the search range
        data_values = np.array(list(map(float, data)))
        
        # Find the top N peaks in each search range
        selected_freqs, selected_powers, selected_stats = find_top_n_peaks(data_values, top_n)

        # If can't find 'top_n' peaks, proceed to next data
        if len(selected_freqs) != top_n:
            return False, None

        stats_flattened = [stat for sublist in selected_stats for stat in sublist]

        # Append data to output
        output_data = (list(selected_freqs) + list(selected_powers) + stats_flattened)

        extracted_features = Helper.extract_feature(output_data)
        scaled_data = sig_scaler.transform(extracted_features)

        new_signature = encoder.predict(scaled_data.reshape(1, -1)).flatten()
        most_similar_camera, similarity_score = Helper.find_most_similar_signature(new_signature, known_signatures)
        is_known = (similarity_score > 0.5) and Helper.is_known_signature(scaled_data, threshold)

        return True, (most_similar_camera, similarity_score, is_known)

    @staticmethod
    def find_most_similar_signature(new_signature, known_signatures):
        similarities = {camera: cosine_similarity([new_signature], sig)[0][0] for camera, sig in known_signatures.items()}
        most_similar_camera = max(similarities, key=similarities.get)
        return most_similar_camera, similarities[most_similar_camera]

    @staticmethod
    def is_known_signature(new_sample, threshold):
        new_sample_encoded = encoder.predict(new_sample.reshape(1, -1))
        new_sample_reconstructed = autoencoder.predict(new_sample.reshape(1, -1))
        new_sample_error = np.mean(np.square(new_sample - new_sample_reconstructed))
        return new_sample_error < threshold

USE_TFLITE_MODEL = True
USE_KERAS_MODEL = False
ESP32_MODEL = False

if USE_TFLITE_MODEL:
    # Load the optimized TensorFlow Lite model
    # model_path = 'models/model_optimize.tflite'
    model_path = 'models/model.tflite'
    # model_path = 'models/dnn_model.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load scaler from file
    scaler = joblib.load('models/scaler.pkl')

    # Function to perform inference on input data
    def perform_inference(input_data):
        # Prepare input data
        input_data = input_data.astype(np.float32)
        # Input data scaled
        input_data_scaled = scaler.fit_transform(input_data.reshape(-1, 1)).reshape(1, -1)
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data_scaled)
        # Run inference
        interpreter.invoke()
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Convert output to binary predictions
        predictions_binary = ((output_data > CAM_THRESHOLD).astype(int))[0]
        confidence = (float(output_data) if predictions_binary[0] == 1 else (1-float(output_data)))*100
        return predictions_binary[0], confidence

elif USE_KERAS_MODEL:
    from tensorflow.keras.models import load_model

    # Load the Keras model
    model_path = 'models/hybrid_cnn_rnn.h5'
    model = load_model(model_path)

    # Load scaler from file
    scaler = joblib.load('models/scaler_cnn.pkl')

    def perform_inference(input_data):
        # Reshape input data to match model input shape
        input_data_reshaped = input_data.reshape(1, -1, 1)
        # Transform the data using the loaded scaler
        input_data_transformed = scaler.transform(input_data_reshaped.reshape(1, -1)).reshape(1, -1, 1)
        # Perform inference
        predictions = model.predict(input_data_transformed)
        # Convert predictions to binary and calculate confidence
        predictions_binary = (predictions >= CAM_THRESHOLD).astype(int)
        confidence = (predictions[0][0] * 100) if predictions_binary[0][0] == 1 else ((1 - predictions[0][0]) * 100)
        prediction = predictions_binary[0][0]

        return prediction, confidence
elif ESP32_MODEL:
    from tensorflow.keras.models import load_model

    # Load the Keras model
    model_path = 'models/hybrid_cnn_rnn_esp.h5'
    model = load_model(model_path)

    # Load scaler from file
    scaler = joblib.load('models/scaler_esp.pkl')

    def perform_inference(input_data):
        # Reshape input data to match model input shape
        input_data_reshaped = input_data.reshape(1, -1, 1)
        # Transform the data using the loaded scaler
        input_data_transformed = scaler.transform(input_data_reshaped.reshape(1, -1)).reshape(1, -1, 1)
        # Perform inference
        predictions = model.predict(input_data_transformed)
        # Convert predictions to binary and calculate confidence
        predictions_binary = (predictions >= CAM_THRESHOLD).astype(int)
        confidence = (predictions[0][0] * 100) if predictions_binary[0][0] == 1 else ((1 - predictions[0][0]) * 100)
        prediction = predictions_binary[0][0]

        return prediction, confidence
else:
    def perform_inference(input_data):
        print('Model not defined')
        return -1, -1

class Analyzer():
    def __init__(self):
        self.rm = None
        self.inst = None
        self.idx = 0

    def setup_analyzer(self, simulated=False):
        if simulated is not None:
            try:
                csv_data = pd.read_csv(simulated)
                # Drop the first column (timestamps)
                csv_data = csv_data.iloc[:, 1:]
                self.inst = csv_data.to_numpy(dtype=float)
            except Exception as e:
                print(f"ERROR: {e}")
        else:
            # Get the VISA resource manager
            self.rm = pyvisa.ResourceManager()
            # Open a session to the Spike software, Spike must be running at this point
            self.inst = self.rm.open_resource('TCPIP0::localhost::5025::SOCKET')
            # For SOCKET programming, we want to tell VISA to use a terminating character
            # to end a read and write operation.
            self.inst.read_termination = '\n'
            self.inst.write_termination = '\n'

            # Set the measurement mode to sweep
            self.inst.write("self.INSTRUMENT:SELECT SA")
            # Configure a 20MHz span sweep at 456MHz
            # Set the RBW/VBW to auto
            # self.inst.write("SENS:BAND:RES:AUTO ON; :BAND:VID:AUTO ON; :BAND:SHAPE FLATTOP")
            self.inst.write("SENS:BAND:RES 500HZ; :BAND:VID 2KHZ; :BAND:SHAPE FLATTOP")
            # Center/span
            self.inst.write("SENS:FREQ:SPAN 1MHZ; CENT 456MHZ")
            # Reference level/Div
            self.inst.write("SENS:POW:RF:RLEV -40DBM; PDIV 10")
            # Set sweep time to 50ms
            self.inst.write("SENS:SWE:TIME 50MS")
            # Peak detector
            self.inst.write("SENS:SWE:DET:FUNC CLEARWRITE; UNIT POWER")
            # Configure the trace. Ensures trace 1 is active and enabled for clear-and-write.
            # These commands are not required to be sent every time; this is for illustrative purposes only.
            self.inst.write("TRAC:SEL 1")  # Select trace 1
            self.inst.write("TRAC:TYPE CLEARWRITE")  # Set clear and write mode
            self.inst.write("TRAC:UPD ON")  # Set update state to on
            self.inst.write("TRAC:DISP ON")  # Set un-hidden

    def capture_data(self, simulated=False):
        if simulated:
            try:
                if self.idx >= len(self.inst):
                    raise IndexError("End of simulated data")
                data = self.inst[self.idx]
                self.idx += 1
            except IndexError:
                return None
            except Exception as e:
                print(f"ERROR: {e}")
                return None
        else:
            # Trigger a sweep, and wait for it to complete
            self.inst.query(":INIT; *OPC?")
            # Sweep data is returned as comma-separated values
            capture = self.inst.query("TRACE:DATA?")  
            data = np.array(capture.split(','), dtype=float)      
        return data

    def teardown(self):
        # Done
        self.inst.close()

def main(args):
    # Extract arguments
    simulated = args.simulated
    model = args.model
    log = args.log

    if model == 'tflite':
        USE_TFLITE_MODEL = True
        USE_KERAS_MODEL = False
    else:
        USE_TFLITE_MODEL = False
        USE_KERAS_MODEL = True

    # Initialize counters for predictions
    consecutive_cam_on = 0
    on_off_predictions = {'cam_off': 0, 'cam_on': 0}

    # Initialize analyzer
    analyzer = Analyzer()
    analyzer.setup_analyzer(simulated)

    if log:        
        # Create a CSV file with timestamp in filename
        filename = f"real_time_SWEEP_REC_{datetime.now().astimezone(tz=None).strftime('%Y-%m-%d %Hh%Mm%Ss')}.csv"
        filepath = os.path.join('output_file', filename)

    with open(filepath, 'w', newline='') as csvfile:
        if log:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'Capture Points', 'Prediction'])

        # Perform sweeps indefinitely every 5 seconds
        while True:
            try:
                # Initialize an empty list to store capture points
                capture_points_list = []

                # Capture data for SAMPLE_COUNT times at each frequency
                for _ in range(SAMPLE_COUNT):
                    # Capture data from analyzer
                    data = analyzer.capture_data(simulated)
                    if data is None:  # Terminate the loop if no more data in simulated mode
                        print("End of simulated data. Exiting...")
                        print(f"Number of predictions=0: {on_off_predictions['cam_off']}")
                        print(f"Number of predictions=1: {on_off_predictions['cam_on']}")
                        for (key, val) in sorted(signature_predictions.items()):
                            print(f'   - [{key}]: {val}')
                        return

                    # Append capture points to the list
                    capture_points_list.append(data)

                # Convert the list to a numpy array
                capture_points_array = np.array(capture_points_list)

                # Get current timestamp with millisecond
                timestamp = Helper.get_current_timestamp()

                # Take the minimum values at each frequency
                capture_points = np.max(capture_points_array, axis=0)
                capture_points_avg = np.average(capture_points_array, axis=0)

                # Perform inference
                prediction, confidence = perform_inference(capture_points)

                if prediction == 0:
                    on_off_predictions['cam_off'] = on_off_predictions.get('cam_off', 0) + 1
                else:
                    on_off_predictions['cam_on'] = on_off_predictions.get('cam_on', 0) + 1
 
                status, ret_val = Helper.find_peak_values(capture_points_avg)
                if status:
                    most_similar_cam, similarity_score, is_known = ret_val[0], ret_val[1], ret_val[2]                                 
                    # Define a threshold for similarity score to classify known/unknown
                    if is_known:
                        signature_predictions[most_similar_cam] = signature_predictions.get(most_similar_cam, 0) + 1
                        print(f"[KNOWN] The new camera data is most similar to the {most_similar_cam} with similarity score: {(100*similarity_score):.3f}%")
                    else:
                        signature_predictions['unknown'] = signature_predictions.get('unknown',0) + 1
                        print(f"[UNKNOWN] The new camera data does not match any known cameras. Similarity score: {similarity_score}")

                # if log:
                #     # Write timestamp, capture points, and prediction to CSV
                #     writer.writerow([timestamp, ','.join(map(str, capture_points)), prediction])
                #     csvfile.flush()

                print(f"Saved data and prediction at {timestamp}, camera: {'OFF' if prediction == 0 else 'ON'} (confidence: {confidence:.3f}%)")

                # # Update prediction counters
                # if prediction == 0:
                #     consecutive_cam_on = 0
                # else:
                #     consecutive_cam_on += 1

                #     # Check if consecutive prediction of 1 is detected 3 times
                #     if consecutive_cam_on >= CAM_ON_LIMIT:
                #         sendEmail()  # Send email notification
                #         print(" >> [ALERT] Email sent to notify user!")
                #         consecutive_cam_on = 0

            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting...")
                print(f"Number of predictions=0: {on_off_predictions['cam_off']}")
                print(f"Number of predictions=1: {on_off_predictions['cam_on']}")
                for (key, val) in sorted(signature_predictions.items()):
                    print(f'   - [{key}]: {val}')
                break

        if not simulated:
            analyzer.teardown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulated', type=str, help='Run in simulated mode with the provided CSV file')
    # parser.add_argument('--simulated', action='store_true', default=False, help='Run in simulated mode. (Default: Real Capture)')
    parser.add_argument('--model', choices=['tflite', 'keras'], default='tflite', help='Specify the model type (default: tflite)')
    parser.add_argument('--log', action='store_true', default=True, help='Option to log output (default: Enabled)')
    args = parser.parse_args()
    main(args)
    sys.exit()