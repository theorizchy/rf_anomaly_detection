import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Step 1: Extract the confusion matrix values
confusion_matrix = np.array([
    [9999,    0,    0,    0],
    [   0, 9106,    0,    0],
    [   0,    0, 9972,    0],
    [   0,  893,   27,    0]
])

# Step 2: Calculate the necessary metrics
# Define the actual and predicted values from the confusion matrix
y_true = np.concatenate([
    np.repeat(0, confusion_matrix[0, 0]),
    np.repeat(1, confusion_matrix[1, 1]),
    np.repeat(2, confusion_matrix[2, 2]),
    np.repeat(3, confusion_matrix[3, 1] + confusion_matrix[3, 2])
])
y_pred = np.concatenate([
    np.repeat(0, confusion_matrix[0, 0]),
    np.repeat(1, confusion_matrix[1, 1]),
    np.repeat(2, confusion_matrix[2, 2]),
    np.concatenate([np.repeat(1, confusion_matrix[3, 1]), np.repeat(2, confusion_matrix[3, 2])])
])

# Compute the classification report
report = classification_report(y_true, y_pred, target_names=[
    'Camera #0 (IMX219 on Raspi 4B)', 
    'Camera #1 (IMX219 on Raspi 4B)', 
    'Camera #2 (OV2640 on ESP-32)', 
    'Unknown Camera'
])
print(report)

# Compute the accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Step 3: Calculate AUC-ROC for multiclass
# Binarize the output labels
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2, 3])

# Compute AUC-ROC for each class and take the average
auc_roc = roc_auc_score(y_true_bin, y_pred_bin, average='macro')
print(f'Macro AUC-ROC: {auc_roc:.4f}')
