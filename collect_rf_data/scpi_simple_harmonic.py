import pyvisa
import time
import os
import csv
from datetime import datetime

NUM_HARMONICS = 3
SAMPLE_COUNT = 10000
NOTE = 'raspi4_b_on' 

def setup_spectrum_analyzer(resource_name):
    # Initialize VISA resource manager and connect to the instrument
    rm = pyvisa.ResourceManager()
    bb = rm.open_resource(resource_name)
    
    # Clear the instrument status and reset
    bb.write('*CLS')
    bb.write('*RST')
    
    # Set read and write terminations
    bb.read_termination = '\n'
    bb.write_termination = '\n'

    # Set up harmonic analysis mode
    bb.write('INSTRUMENT:SELECT HARM')

    # Set center frequency, span, and RBW/VBW
    bb.write("SENS:BAND:RES 1KHZ; :BAND:VID 500HZ")
    bb.write("SENS:FREQ:SPAN 200KHZ; CENT 456MHZ")
    bb.write("SENS:SWE:TIME 50MS")
    # bb.write(":TRAC:TYPE CLEARWRITE")

    # Set harmonic count
    bb.write(f'HARM:NUMB {NUM_HARMONICS}')
    bb.write('HARM:TRACK ON')
    bb.write('HARM:MODE PEAK')
    bb.write('HARM:FREQ:FUND 456MHZ')
    bb.write('HARM:FREQ:SPAN 200KHZ')
    # bb.write('HARM:TRAC:TYPE MAXHOLD')
    bb.write('HARM:TRAC:TYPE WRITES')
    # bb.write('HARM:BAND:RES 1KHZ; :BAND:VID 500HZ')
    
    return bb

def capture_data(bb, num_points=100):
    # Capture amplitude and delta data
    data_points = []

    for _ in range(num_points):
        # Trigger a sweep and wait for it to complete
        bb.query(":INIT; *OPC?")

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        frequencies = []
        amplitudes = []

        for i in range(NUM_HARMONICS):   
            # Fetch frequency and amplitude
            freq = float(bb.query(f":FETC:HARM:FREQ? {i+1}").strip())
            amp = float(bb.query(f":FETC:HARM:AMPL? {i+1}").strip())

            frequencies.append(freq)
            amplitudes.append(amp)

        # Store the data points as a single row
        data_points.append((timestamp, frequencies, amplitudes))
        row = [timestamp] + [f"{freq / 1e6:.3f}" for freq in frequencies] + [f"{amp:.3f}" for amp in amplitudes]
        print(f"{' '.join([f'{cell:<20}' for cell in row])}")

    return data_points

def main():
    resource_name = 'TCPIP0::localhost::5025::SOCKET'
    bb = setup_spectrum_analyzer(resource_name)
    
    print("Capturing data...")
    # Print captured data in table format
    header = ["Timestamp"] + [f"freq_{i+1}_MHz" for i in range(NUM_HARMONICS)] + [f"amp_{i+1}_dBm" for i in range(NUM_HARMONICS)]
    print(f"{' '.join([f'{col:<20}' for col in header])}")
    print("="*len(header)*20)

    # Start capturing data
    data_points = capture_data(bb, SAMPLE_COUNT)

    # Optionally save to CSV file
    filename = f"harmonics_clearwrite_captured_REC_{datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')}_{NOTE}.csv"
    file = os.path.join('output_file', 'harmonics', filename)
    
    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for timestamp, freqs, amps in data_points:
            row = [timestamp] + [round(freq / 1e6, 3) for freq in freqs] + [round(amp, 3) for amp in amps]
            writer.writerow(row)

    # Close the instrument connection
    bb.close()

if __name__ == "__main__":
    main()
