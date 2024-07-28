import os
import csv
import pyvisa
from datetime import datetime

distance = 'esp32_on'
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
# inst.write("SENS:BAND:RES AUTO ON; :BAND:VID AUTO ON")
# Set the RBW to 500Hz and VBW to 2KHz
inst.write("SENS:BAND:RES 500HZ; :BAND:VID 2KHZ")
# Set the band to flattop
inst.write("SENS:BAND:SHAPE FLATTOP")
# Center/span, configure a 1MHz span sweep at 456MHz
inst.write("SENS:FREQ:SPAN 1MHZ; CENT 12MHZ")
# Reference level/Div
inst.write("SENS:POW:RF:RLEV -40DBM; PDIV 10")
# Set sweep time to 50ms
inst.write("SENS:SWE:TIME 50MS")
# Clear and write mode
inst.write("SENS:SWE:DET:FUNC CLEARWRITE; UNIT POWER")

# Configure the trace. Ensures trace 1 is active and enabled for clear-and-write.
# These commands are not required to be sent every time; this is for illustrative purposes only.
inst.write("TRAC:SEL 1")  # Select trace 1
inst.write("TRAC:TYPE CLEARWRITE")  # Set clear and write mode
inst.write("TRAC:UPD ON")  # Set update state to on
inst.write("TRAC:DISP ON")  # Set un-hidden

# Get current timestamp with millisecond
def get_current_timestamp():
    return datetime.now().astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S.%f')

# Create a CSV file with timestamp in filename
filename = f"clearwrite_captured_SWEEP_REC_{datetime.now().astimezone(tz=None).strftime('%Y-%m-%d %Hh%Mm%Ss')}_{distance}.csv"
filepath = os.path.join('output_file', filename)
with open(filepath, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Perform 10000 sweeps
    for i in range(10000):
        # Get current timestamp with millisecond
        timestamp = get_current_timestamp()

        # Trigger a sweep, and wait for it to complete
        inst.query(":INIT; *OPC?")

        # Sweep data is returned as comma-separated values
        data = inst.query("TRACE:DATA?")

        # Write timestamp and capture points to CSV
        writer.writerow([timestamp] + data.split(','))
        print(f"Saved data for sweep {i+1}")

# Done
inst.close()
