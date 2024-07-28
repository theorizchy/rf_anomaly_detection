import pandas as pd
import numpy as np
from scipy.signal import correlate

# Load the data
file_path = 'sample_150cm.txt'
data = pd.read_csv(file_path, header=None)

# Extract timestamps and power measurements
timestamps = data.iloc[:, 0]
power_measurements = data.iloc[:, 1:]

# Function to calculate autocorrelation using scipy
def autocorrelation(x):
    result = correlate(x, x, mode='full', method='fft')
    return result[result.size // 2:]

# Calculate autocorrelation for each sample
autocorrelations = power_measurements.apply(lambda row: autocorrelation(row.values), axis=1)

# Display the autocorrelation values
print(autocorrelations)
