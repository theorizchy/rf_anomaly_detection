import os
import csv
import struct
import numpy as np
from enum import Enum
from datetime import datetime

SHR_FILE_SIGNATURE = 0xAA10
SHR_FILE_VERSION = 0x2

SHR_SCALE_DBM = 0
SHR_SCALE_MV = 1

class ShrWindow(Enum):
    SHR_WINDOW_NUTTALL = 0
    SHR_WINDOW_FLATTOP = 1
    SHR_WINDOW_GAUSSIAN = 2

class ShrDecimationType(Enum):        
    SHR_DECIMATION_TYPE_TIME = 0
    SHR_DECIMATION_TYPE_COUNT = 1

class ShrDecimationDetector(Enum):        
    SHR_DECIMATION_DETECTOR_AVG = 0
    SHR_DECIMATION_DETECTOR_MAX = 1

class ShrChannelizerOutput(Enum):        
    SHR_CHANNELIZER_OUTPUT_UNITS_DBM = 0
    SHR_CHANNELIZER_OUTPUT_UNITS_DBMHZ = 1

class ShrVideoDetector(Enum):        
    SHR_VIDEO_DETECTOR_MINMAX = 0
    SHR_VIDEO_DETECTOR_AVG = 1

class ProcessingUnits(Enum):
    SHR_VIDEO_UNITS_LOG = 0
    SHR_VIDEO_UNITS_VOLTAGE = 1
    SHR_VIDEO_UNITS_POWER = 2
    SHR_VIDEO_UNITS_SAMPLE = 3

# 472 Bytes
SHR_FILE_HEADER_FORMAT = "<HHIQIIdd128HddddfIfIiiiidiiiiiidd16I"
# 48 Bytes
SHR_SWEEP_HEADER_FORMAT = "<QdddB15B"

def parse_shr_file(file_path):
    # Dictionary to store sweep data
    sweep_data_dict = {}
    
    with open(file_path, "rb") as f:
        # Read file header
        file_header_data = f.read(struct.calcsize(SHR_FILE_HEADER_FORMAT))
        file_header = struct.unpack(SHR_FILE_HEADER_FORMAT, file_header_data)

        # Verify file signature and version
        signature, version, reserved1, data_offset, sweep_count, sweep_length, first_bin_freq_hz, bin_size_hz, *remaining_data = file_header
        # Continue decoding the remaining variables
        center_freq_hz, span_hz, rbw_hz, vbw_hz, ref_level, ref_scale, x, window, attenuation, gain, detector, processing_units, window_bandwidth, decimation_type, decimation_detector, decimation_count, decimation_time_ms, channelize_enabled, channel_output_units, channel_center_hz, channel_width_hz, *_ = remaining_data[128:]
       
        if signature != SHR_FILE_SIGNATURE or version > SHR_FILE_VERSION:
            raise ValueError("Invalid file signature or version")

        print(f"File Name: {file_path}")
        print(f"Signature: {'OK' if signature == SHR_FILE_SIGNATURE else 'NOK'}")
        print(f"SHR Version: {version}")
        print(f"Sweep Count: {sweep_count}")
        print(f"Sweep Size: {sweep_length}")
        print(f"Sweep Start Freq: {first_bin_freq_hz}")
        print(f"Sweep Bin Size: {bin_size_hz}")
        print(f"Sweep Freq Range: {(center_freq_hz - span_hz / 2.0) * 1.0e-3} kHz to {(center_freq_hz + span_hz / 2.0) * 1.0e-9} GHz")
        print(f"RBW: {rbw_hz * 1.0e-3} kHz")
        print(f"VBW: {vbw_hz * 1.0e-3} kHz")
        print(f"Reference Level: {ref_level} {'dBm' if ref_scale == SHR_SCALE_DBM else 'mV'}")
        print(f"Div: {x}")
        print(f"Window: \"{ShrWindow(window).name}\"")
        print(f"Attenuation: {attenuation}")
        print(f"Gain: {gain}")
        print(f"Detector: \"{ShrVideoDetector(detector).name}\"")
        print(f"Processing Units: \"{ProcessingUnits(processing_units).name}\"")
        print(f"Window Bandwidth: {window_bandwidth} bins")
        print(f"Decimation Type: \"{ShrDecimationType(decimation_type).name}\"")
        print(f"Decimation Detector: \"{ShrDecimationDetector(decimation_detector).name}\"")
        print(f"Decimation Count: {decimation_count}")
        print(f"Decimation Time (ms): {decimation_time_ms}")
        print(f"Channelize Enabled: {channelize_enabled}")
        print(f"Channel Output units: \"{ShrChannelizerOutput(channel_output_units).name}\"")
        print(f"Channel Center: {channel_center_hz * 1.0e-9} GHz")
        print(f"Channel Width: {channel_width_hz *1.0e-6} MHz\n")

        # Read sweeps
        for i in range(sweep_count):
        # for i in range(sweep_count):
            # Read sweep header
            sweep_header_data = f.read(struct.calcsize(SHR_SWEEP_HEADER_FORMAT))
            sweep_header = struct.unpack(SHR_SWEEP_HEADER_FORMAT, sweep_header_data)

            timestamp, latitude, longitude, altitude, adc_overflow, *reserved2 = sweep_header
            # Convert milliseconds to seconds
            epoch_time_seconds = timestamp / 1000

            # print(f"Sweep {i}:")
            dt_object = datetime.fromtimestamp(epoch_time_seconds)
            formatted_timestamp = dt_object.strftime('%Y-%m-%d %H:%M:%S.%f')
            # print(f"Date/Time: {formatted_timestamp}")
            # print(f"Latitude: {latitude}")
            # print(f"Longitude: {longitude}")
            # print(f"Altitude: {altitude}")
            # print(f"ADC Overflow: {adc_overflow}")

            # Read sweep data
            sweep_data = np.fromfile(f, dtype=np.float32, count=sweep_length)
            # Store sweep data in dictionary
            sweep_data_dict[formatted_timestamp] = sweep_data

        return sweep_data_dict


def save_parsed_shr_file(sweep_data_dict={}, output_file='sweep_data'):
    # Save dictionary as CSV file
    csv_file_name = 'parsed_' + output_file + '.csv'
    csv_file_path = os.path.join('output_file', csv_file_name)
    with open(csv_file_path, "w", newline='') as csvfile:
        # Create a CSV writer object associated with the file
        writer = csv.writer(csvfile)
        # Write data row by row
        for (timestamp, v) in sweep_data_dict.items():
            # Format timestamp to limit decimal places to 3
            # Format data to limit decimal places to 3
            data = ["{:.3f}".format(value) for value in v.tolist()]
            row_data = [timestamp] + data
            # Write the row to the CSV file
            writer.writerow(row_data)


if __name__ == "__main__":
    filename = "SWEEP_REC_2024-03-12 21h36m56s_cam_on.shr"
    # filename = "SWEEP_REC_2024-03-12 21h02m38s_cam_on.shr"
    # filename = "SWEEP_REC_2024-03-12 20h42m01s_cam_off.shr"
    # filename = "SWEEP_REC_2024-03-12 20h27m34s_cam_on.shr"
    file_path = os.path.join('input_file', filename)
    parsed_dict = parse_shr_file(file_path)
    save_parsed_shr_file(parsed_dict, filename[:-4])
