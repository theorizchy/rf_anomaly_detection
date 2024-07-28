import os
import csv
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from keras.models import load_model

# Function to find the highest peak in the given range
def find_peak(measurements, indices, start_freq, interval):
    search_range = measurements[indices]
    max_index = np.argmax(search_range)
    peak_index = indices.start + max_index
    peak_freq = start_freq + peak_index * interval
    peak_pwr = search_range[max_index]
    return peak_freq, peak_pwr

def find_peak_values(file_path, start_freq, stop_freq, center_freq, delta=50000, top_n=5, plot=False):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)

        # Read the first row to determine the number of columns
        first_row = next(reader)
        num_columns = len(first_row)
        
        # Calculate the interval
        num_points = num_columns - 1  # Exclude the timestamp column
        interval = (stop_freq - start_freq) / num_points

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

        # Initialize output data list
        output_data = [['timestamp'] + [f'peak_{i+1}_freq' for i in range(top_n)] + [f'peak_{i+1}_pwr' for i in range(top_n)] +
                       [f'var_interval_{i+1}' for i in range(top_n)] + [f'skewness_interval_{i+1}' for i in range(top_n)] + [f'kurtosis_interval_{i+1}' for i in range(top_n)]]

        # Print header
        header = "timestamp     " + "   ".join([f"peak_{i+1}_freq" for i in range(top_n)]) + "   " + "   ".join([f"peak_{i+1}_pwr" for i in range(top_n)]) + \
                 "   " + "   ".join([f"var_interval_{i+1}" for i in range(top_n)]) + "   " + "   ".join([f"skewness_interval_{i+1}" for i in range(top_n)]) + "   " + "   ".join([f"kurtosis_interval_{i+1}" for i in range(top_n)])
        print(header)
        print("=" * len(header))

        cur_row = 0
        # max_row = 1

        # Process the remaining rows
        for row in reader:
            # if cur_row >= max_row:
            #     break
            cur_row += 1
            # Convert values to float and extract the search range
            row_values = np.array(list(map(float, row[1:])))
            
            # Find the top N peaks in each search range
            selected_freqs, selected_powers, selected_stats = find_top_n_peaks(row_values, top_n)

            # If can't find 'top_n' peaks, proceed to next data
            if len(selected_freqs) != top_n:
                continue

            stats_flattened = [stat for sublist in selected_stats for stat in sublist]

            # Append data to output
            output_data.append([row[0]] + list(selected_freqs) + list(selected_powers) + stats_flattened)

            # Print the row data
            row_str = f"{row[0]:<13}  " + "  ".join([f"{freq:<12.3f}  {pwr:<8.3f}" for freq, pwr in zip(selected_freqs, selected_powers)]) + \
                      "  " + "  ".join([f"{stat:<12.3f}" for stat in stats_flattened])
            print(row_str)

            if plot and (cur_row % 1000 == 0):
                frequencies = np.linspace(start_freq, stop_freq, num_points)
                plt.figure(figsize=(10, 6))
                plt.plot(frequencies, row_values, label='Spectrum Data')
                
                for i, (freq, pwr) in enumerate(zip(selected_freqs, selected_powers)):
                    plt.plot(freq, pwr, 'o', label=f'Peak {i+1}: {freq/1e6:.2f} MHz')
                
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power (dBm)')
                plt.title(f'Spectrum Analysis for {row[0]}')
                plt.legend()
                plt.grid(True)
                plt.show()

        # Save to CSV
        output_folder = 'extract_peaks'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        output_file = os.path.join(output_folder, os.path.basename(file_path))
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(output_data)

        print(f"Extracted peaks saved to: {output_file}")

# Usage
filename = [
    "clearwrite_captured_SWEEP_REC_2024-07-07 15h57m07s_raspi4_b_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 16h31m15s_raspi4_b_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 18h42m08s_raspi4_a_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 19h17m56s_raspi4_a_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 14h44m03s_esp32_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-13 12h20m09s_esp32_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-13 13h34m11s_esp32_on_led.csv"
]

folder_name = 'output_file'

for f in filename:
    print(f"\nProcessing file: {f}")
    if 'esp32' in f:
        start_freq = 11500000   # 11.5 MHz 
        center_freq = 12000000  # 12 MHz
        stop_freq = 12500000    # 12.5 MHz
    elif 'raspi' in f:
        start_freq = 455500000   # 455.5 MHz 
        center_freq = 456000000  # 456 MHz
        stop_freq = 456500000    # 456.5 MHz
    find_peak_values(os.path.join(folder_name, f), start_freq, stop_freq, center_freq, top_n=5)
