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

SEED = 42

X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_val   = np.load(X_VAL_PATH)
y_val   = np.load(Y_VAL_PATH)
X_test  = np.load(X_TEST_PATH)
y_test  = np.load(Y_TEST_PATH)

X_train = X_train.astype(np.float32, copy=False)
X_val   = X_val.astype(np.float32, copy=False)
X_test  = X_test.astype(np.float32, copy=False)
y_train = y_train.astype(np.int32,   copy=False)
y_val   = y_val.astype(np.int32,     copy=False)
y_test  = y_test.astype(np.int32,    copy=False)

#------------- Print Distribution -------------

def distribution_sum(y, name):
    uniq, cnt = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
    total = int(y.shape[0])
    print(f"📈 {name}: {dist} (total={total:,})")

distribution_sum(y_train, "Train")
distribution_sum(y_val,   "Validation")
distribution_sum(y_test,  "Test")
print("/n")

#-----------  Imbalance Adjustment ----------- 
#scale_pos_weight = neg / pos

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = float(n_neg) / max(float(n_pos), 1.0)
print(f"scale_pos_weight: {scale_pos_weight:.3f}  (neg={n_neg:,} / pos={n_pos:,})\n")

#--------- Define and train the model ---------

xgb = XGBClassifier(
    n_estimators=1000, #maximum trees
    max_depth=6, #depth of the trees
    learning_rate=0.08, #boosting learning rate
    subsample=0.8, # sampling
    colsample_bytree=0.8,#sampling
    reg_lambda=1.0, #regularization to prevent overfitting
    reg_alpha=0.0, #regularization to prevent overfitting
    random_state=SEED, # Set a fixed random seed
    n_jobs=-1, #Set the number of cores
    tree_method="gpu_hist", #Set the use of gpu instead cpu
    scale_pos_weight=scale_pos_weight, #adjust the weight of the minority class
    use_label_encoder=False, #Prevents XGBoost from using the internal label encode           
    eval_metric="auc" #Area Under the ROC Curve (Receiver Operating Characteristic)                 
)