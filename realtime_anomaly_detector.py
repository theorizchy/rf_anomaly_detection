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

# Limit for consecutive camera ON
CAM_ON_LIMIT = 50
# Prediction threshold for camera ON
CAM_THRESHOLD = 0.5
# Numbers of sample to be taken before actual prediction
SAMPLE_COUNT = 100
# Majority Threshold
MAJORITY_THRESHOLD = 0.5


class Helper:
    # Get current timestamp with millisecond
    @staticmethod
    def get_current_timestamp():
        return datetime.now().astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S.%f')


USE_TFLITE_MODEL = True
USE_KERAS_MODEL = False

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
    scaler = joblib.load('models/scaler.pkl')

    def perform_inference(input_data):
        # Reshape input data to match model input shape
        input_data_reshaped = input_data.reshape(1, -1, 1, 1)
        # Transform the data using the loaded scaler
        input_data_transformed = scaler.transform(input_data_reshaped.reshape(1, -1)).reshape(1, -1, 1, 1)
        # Perform inference
        predictions = model.predict(input_data_transformed)
        # Apply majority voting for each timestep
        binary_predictions = (predictions >= CAM_THRESHOLD).astype(int)
        majority_votes = np.mean(binary_predictions, axis=0)  # Compute mean along the batch axis
        # Calculate confidence (percentage of 1s)
        confidence = np.mean(majority_votes) * 100
        # Predict ON if >50% confidence
        prediction = 1 if confidence > CAM_THRESHOLD else 0
        # Invert confidence for camera OFF
        confidence = (100-confidence) if prediction == 0 else confidence

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
            self.inst.write("SENS:BAND:RES:AUTO ON; :BAND:VID:AUTO ON; :BAND:SHAPE FLATTOP")
            # Center/span
            self.inst.write("SENS:FREQ:SPAN 1MHZ; CENT 456MHZ")
            # Reference level/Div
            self.inst.write("SENS:POW:RF:RLEV -40DBM; PDIV 10")
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

    # Initialize counters for predictions
    cam_off_count = 0
    cam_on_count = 0
    consecutive_cam_on = 0

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
                        print(f"Number of predictions=0: {cam_off_count}")
                        print(f"Number of predictions=1: {cam_on_count}")
                        return

                    # Append capture points to the list
                    capture_points_list.append(data)

                # Convert the list to a numpy array
                capture_points_array = np.array(capture_points_list)

                # Get current timestamp with millisecond
                timestamp = Helper.get_current_timestamp()

                # Take the minimum values at each frequency
                min_capture_points = np.max(capture_points_array, axis=0)

                # Perform inference
                prediction, confidence = perform_inference(min_capture_points)

                if log:
                    # Write timestamp, capture points, and prediction to CSV
                    writer.writerow([timestamp, ','.join(map(str, min_capture_points)), prediction])
                    csvfile.flush()

                print(f"Saved data and prediction at {timestamp}, camera: {'OFF' if prediction == 0 else 'ON'} (confidence: {confidence:.3f}%)")

                # Update prediction counters
                if prediction == 0:
                    cam_off_count += 1
                    consecutive_cam_on = 0
                else:
                    cam_on_count += 1
                    consecutive_cam_on += 1

                    # Check if consecutive prediction of 1 is detected 3 times
                    if consecutive_cam_on >= CAM_ON_LIMIT:
                        sendEmail()  # Send email notification
                        print(" >> [ALERT] Email sent to notify user!")
                        consecutive_cam_on = 0

            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting...")
                print(f"Number of predictions=0: {cam_off_count}")
                print(f"Number of predictions=1: {cam_on_count}")
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