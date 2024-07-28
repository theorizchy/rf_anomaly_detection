import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import OneClassSVM
import joblib

def preprocess_data(file):
    data = pd.read_csv(file)
    selected_data = data.drop(columns=['timestamp'])
    normalized_data = pd.DataFrame(selected_data, columns=data.columns[1:])
    return normalized_data

folder_name = 'extract_feature'

# Usage
filename = [
    # "clearwrite_captured_SWEEP_REC_2024-07-07 15h57m07s_raspi4_b_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 16h31m15s_raspi4_b_on.csv",
    # "clearwrite_captured_SWEEP_REC_2024-07-07 18h42m08s_raspi4_a_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 19h17m56s_raspi4_a_on.csv",
    "clearwrite_captured_SWEEP_REC_2024-07-07 14h44m03s_esp32_on.csv",
]

data = []
for f in filename:
    print(f"\nProcessing file: {f}")
    data.append(preprocess_data(os.path.join(folder_name, f)))

X = np.concatenate(data)
y = np.concatenate([np.ones(9999)*0, np.ones(9999)*1, np.ones(9999)*2])  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Classifier
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
joblib.dump(clf_rf, 'signature/model/rf_cam_classifier_model.pkl')

# Train XGBoost Classifier
clf_xgb = xgb.XGBClassifier(
    objective='multi:softmax',  # Adjust based on number of classes (multi:softmax for multi-class)
    num_class=3,  # Number of classes (camera IDs)
    n_estimators=100,  # Number of trees (boosting rounds)
    random_state=42
)
clf_xgb.fit(X_train, y_train)
y_pred_xgb = clf_xgb.predict(X_test)
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
joblib.dump(clf_xgb, 'signature/model/xgb_cam_classifier_model.pkl')

# Train OneClassSVM for known/unknown detection
clf_ocsvm = OneClassSVM(gamma='auto', nu=0.05)  # nu is the proportion of outliers expected
clf_ocsvm.fit(X_train)
joblib.dump(clf_ocsvm, 'signature/model/ocsvm_cam_classifier_model.pkl')


for f in filename:
    X_new = preprocess_data(os.path.join(folder_name, f))
    X_new_values = X_new.values

    y_pred_rf = clf_rf.predict(X_new_values)
    y_pred_xgb = clf_xgb.predict(X_new_values)
    y_pred_ocsvm = clf_ocsvm.predict(X_new_values)

    print(f"\nFor file: {f}")
    print("Random Forest Predictions:")
    unique, counts = np.unique(y_pred_rf, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"- Camera ID {int(cls)}: {count}")
    
    print("XGBoost Predictions:")
    unique, counts = np.unique(y_pred_xgb, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"- Camera ID {int(cls)}: {count}")
    
    print("OneClassSVM Predictions (1 for known, -1 for unknown):")
    unique, counts = np.unique(y_pred_ocsvm, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"- {'Known' if cls == 1 else 'Unknown'}: {count}")