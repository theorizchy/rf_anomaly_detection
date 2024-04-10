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

# Limit for consecutive camera ON
CAM_ON_LIMIT = 5

# Initialize counters for predictions
cam_off_count = 0
cam_on_count = 0
consecutive_cam_on = 0

# Load the optimized TensorFlow Lite model
optimized_model_path = 'models/model_optimize.tflite'
interpreter = tf.lite.Interpreter(model_path=optimized_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get the VISA resource manager
rm = pyvisa.ResourceManager()

# Open a session to the Spike software, Spike must be running at this point
inst = rm.open_resource('TCPIP0::localhost::5025::SOCKET')

# For SOCKET programming, we want to tell VISA to use a terminating character
# to end a read and write operation.
inst.read_termination = '\n'
inst.write_termination = '\n'

# Set the measurement mode to sweep
inst.write("INSTRUMENT:SELECT SA")

# Configure a 20MHz span sweep at 1GHz
# Set the RBW/VBW to auto
inst.write("SENS:BAND:RES:AUTO ON; :BAND:VID:AUTO ON; :BAND:SHAPE FLATTOP")
# Center/span
inst.write("SENS:FREQ:SPAN 6GHZ; CENT 3GHZ")
# Reference level/Div
inst.write("SENS:POW:RF:RLEV -20DBM; PDIV 10")
# Peak detector
inst.write("SENS:SWE:DET:FUNC MINMAX; UNIT POWER")

# Configure the trace. Ensures trace 1 is active and enabled for clear-and-write.
# These commands are not required to be sent every time; this is for illustrative purposes only.
inst.write("TRAC:SEL 1")  # Select trace 1
inst.write("TRAC:TYPE WRITE")  # Set clear and write mode
inst.write("TRAC:UPD ON")  # Set update state to on
inst.write("TRAC:DISP ON")  # Set un-hidden

# Function to perform inference on input data
def perform_inference(input_data):
    # Prepare input data
    input_data = input_data.astype(np.float32)
    # Standard scaling
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data.reshape(-1, 1)).reshape(1, -1)
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data_scaled)
    # Run inference
    interpreter.invoke()
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Convert output to binary predictions
    predictions_binary = (output_data > 0.5).astype(int)
    return predictions_binary[0]

# Get current timestamp with millisecond
def get_current_timestamp():
    return datetime.now().astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S.%f')

# Create a CSV file with timestamp in filename
filename = f"real_time_SWEEP_REC_{datetime.now().astimezone(tz=None).strftime('%Y-%m-%d %Hh%Mm%Ss')}.csv"
filepath = os.path.join('output_file', filename)

with open(filepath, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Timestamp', 'Capture Points', 'Prediction'])

    # Perform sweeps indefinitely every 5 seconds
    while True:
        try:
            # Get current timestamp with millisecond
            timestamp = get_current_timestamp()

            # Trigger a sweep, and wait for it to complete
            inst.query(":INIT; *OPC?")

            # Sweep data is returned as comma-separated values
            data = inst.query("TRACE:DATA?")

            # Convert data to numpy array
            capture_points = np.array(data.split(','), dtype=float)

            # Perform inference
            prediction = perform_inference(capture_points)[0]

            # Write timestamp, capture points, and prediction to CSV
            writer.writerow([timestamp, data, prediction])
            csvfile.flush()

            print(f"Saved data and prediction at {timestamp}, camera: {'OFF' if prediction == 0 else 'ON'}")

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

            # Wait for 5 seconds before the next sweep
            time.sleep(5)

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Exiting...")
            print(f"Number of predictions=0: {cam_off_count}")
            print(f"Number of predictions=1: {cam_on_count}")
            break

# Done
inst.close()