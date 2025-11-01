import os
import json
import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, classification_report
)
from joblib import dump
from datetime import datetime
import platform

#------------------- FILES -------------------

# Bring path numpy 
BASE_DIR = r"numpy"

#Bring files
X_TRAIN_PATH = os.path.join(BASE_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(BASE_DIR, "y_train.npy")
X_VAL_PATH   = os.path.join(BASE_DIR, "X_val.npy")
Y_VAL_PATH   = os.path.join(BASE_DIR, "y_val.npy")
X_TEST_PATH  = os.path.join(BASE_DIR, "X_test.npy")
Y_TEST_PATH  = os.path.join(BASE_DIR, "y_test.npy")

#Result files
MODEL_PATH   = os.path.join(BASE_DIR, "model_xgb_ids.pkl")
METADATA_JSON = os.path.join(BASE_DIR, "model_xgb_ids_metadata.json")

