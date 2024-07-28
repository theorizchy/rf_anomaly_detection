import os
import pandas as pd
import csv

def extract_feature(file):
    data = pd.read_csv(file)

    # Calculate the ratios of the powers
    data['ratio_2_to_1'] = data['peak_2_pwr'] / data['peak_1_pwr']
    data['ratio_2_to_3'] = data['peak_2_pwr'] / data['peak_3_pwr']
    
    # Relative power strength from total
    sum_peak_powers = data[['peak_1_pwr', 'peak_2_pwr', 'peak_3_pwr']].sum(axis=1)
    # Normalize the peak powers
    data['norm_peak_1_pwr'] = data['peak_1_pwr'] / sum_peak_powers
    data['norm_peak_2_pwr'] = data['peak_2_pwr'] / sum_peak_powers
    data['norm_peak_3_pwr'] = data['peak_3_pwr'] / sum_peak_powers

    # Calculate the widths between the frequencies
    span_freq = data['peak_3_freq'] - data['peak_1_freq']
    width_1_to_2 = data['peak_2_freq'] - data['peak_1_freq']
    width_2_to_3 = data['peak_3_freq'] - data['peak_2_freq']
    # Normalize the widths by the overall span of the frequencies
    data['rel_width_1_to_2'] = width_1_to_2/span_freq
    data['rel_width_2_to_3'] = width_2_to_3/span_freq


    # Save to CSV
    output_folder = 'extract_feature'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_file = os.path.join(output_folder, os.path.basename(file))

    # Select only the newly created features
    new_features = data[['timestamp', 'ratio_2_to_1', 'ratio_2_to_3', 'peak_1_freq', 'peak_2_freq', 'peak_3_freq',
                        'norm_peak_1_pwr', 'norm_peak_2_pwr', 'norm_peak_3_pwr',
                        'rel_width_1_to_2', 'rel_width_2_to_3']]

    # Save the new features to a new CSV file
    new_features.to_csv(output_file, index=False)

    print(f"Extracted peaks saved to: {output_file}")
    pass

folder_name = 'extract_peaks'

# Usage
filename = [
    "clearwrite_captured_SWEEP_REC_2024-07-07 15h57m07s_raspi4_b_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 16h31m15s_raspi4_b_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 18h42m08s_raspi4_a_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 19h17m56s_raspi4_a_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 14h44m03s_esp32_on.csv",
]

for f in filename:
    print(f"\nProcessing file: {f}")
    extract_feature(os.path.join(folder_name, f))
