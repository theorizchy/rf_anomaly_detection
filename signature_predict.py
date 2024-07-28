import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(file):
    data = pd.read_csv(file)
    selected_data = data.drop(columns=['timestamp'])
    normalized_data = pd.DataFrame(selected_data, columns=data.columns[1:])
    return normalized_data

# Load the trained classifier model
clf = joblib.load('signature/model/rf_cam_classifier_model.pkl')

filename = [
    "clearwrite_captured_SWEEP_REC_2024-07-07 15h57m07s_raspi4_b_on.csv",
    # "clearwrite_captured_SWEEP_REC_2024-07-07 16h31m15s_raspi4_b_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 18h42m08s_raspi4_a_on.csv",
    # "clearwrite_captured_SWEEP_REC_2024-07-07 19h17m56s_raspi4_a_on.csv",
    # "clearwrite_captured_SWEEP_REC_2024-07-07 14h44m03s_esp32_on.csv",
]

for f in filename:
    X_new = preprocess_data(os.path.join('extract_feature',f))

    # Make predictions using the loaded model
    predictions = clf.predict(X_new.values)
    # Count occurrences of each camera ID
    prediction_counts = pd.Series(predictions).value_counts()

    # Print the counts
    print(f"For file: {f}")
    for camera_id, count in prediction_counts.items():
        print(f"- Camera ID {int(camera_id)}: {count}")
    print()