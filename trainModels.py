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
from xgboost.callback import EarlyStopping

#------------------- FILES -------------------

# Bring path numpy 
BASE_DIR = r"numpy"

#Bring files
X_TRAIN_PATH = os.path.join(BASE_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(BASE_DIR, "y_train.npy")
X_VAL_PATH = os.path.join(BASE_DIR, "X_val.npy")
Y_VAL_PATH = os.path.join(BASE_DIR, "y_val.npy")
X_TEST_PATH = os.path.join(BASE_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(BASE_DIR, "y_test.npy")

#Result files
MODEL_PATH   = os.path.join(BASE_DIR, "model_xgb_ids.pkl")
METADATA_JSON = os.path.join(BASE_DIR, "model_xgb_ids_metadata.json")

SEED = 42

X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_val = np.load(X_VAL_PATH)
y_val = np.load(Y_VAL_PATH)
X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

X_train = X_train.astype(np.float32, copy=False)
X_val = X_val.astype(np.float32, copy=False)
X_test = X_test.astype(np.float32, copy=False)
y_train = y_train.astype(np.int32, copy=False)
y_val = y_val.astype(np.int32, copy=False)
y_test = y_test.astype(np.int32, copy=False)

#------------- Print Distribution -------------

def distribution_sum(y, name):
    uniq, cnt = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
    total = int(y.shape[0])
    print(f" {name}: {dist} (total={total:,})")

distribution_sum(y_train,"Train")
distribution_sum(y_val,"Validation")
distribution_sum(y_test,"Test")
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
    tree_method="hist", #Set the use of cpu instead gpu
    scale_pos_weight=scale_pos_weight, #adjust the weight of the minority class
    use_label_encoder=False, #Prevents XGBoost from using the internal label encode           
    eval_metric="auc" #Area Under the ROC Curve (Receiver Operating Characteristic)                 
)

t0 = time.time()

xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)


train_time = time.time() - t0
print(f"Train time: {train_time:.1f}s")
print(f"Used trees (best_iteration): {xgb.best_iteration + 1 if xgb.best_iteration is not None else 'n/a'}\n")

#--------- Choose the best threshold ---------

val_prob = xgb.predict_proba(X_val)[:, 1]

#Sweep possible thresholds from the Precision-Recall curve
prec, rec, thr = precision_recall_curve(y_val, val_prob)

#Avoids division by zero and selects the threshold with the highest F1 score
f1_scores = (2 * prec * rec) / np.clip(prec + rec, a_min=1e-12, a_max=None)
best_idx = int(np.argmax(f1_scores))
best_threshold = 0.5 if best_idx >= len(thr) else float(thr[best_idx])

print(f"Best threshold (validation, max F1): {best_threshold:.6f}")
print(f"F1(val)={f1_scores[best_idx]:.4f} | Prec(val)={prec[best_idx]:.4f} | Rec(val)={rec[best_idx]:.4f}\n")

# --------- Evaluate on the test set ---------

test_prob = xgb.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

acc = accuracy_score(y_test, test_pred)
pre = precision_score(y_test, test_pred, zero_division=0)
rec = recall_score(y_test, test_pred, zero_division=0)
f1  = f1_score(y_test, test_pred, zero_division=0)
auc = roc_auc_score(y_test, test_prob)
ap  = average_precision_score(y_test, test_prob)  # AUC-PR

cm = confusion_matrix(y_test, test_pred)
tn, fp, fn, tp = cm.ravel()

print("\n------- Results -------")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (P): {pre:.4f}")
print(f"Recall (R): {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"AUC-PR (AP): {ap:.4f}")
print("\nConfusion Matrix [TN FP; FN TP]:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, test_pred, digits=4, zero_division=0))

#---------- Save model and metadata ----------

dump(xgb, MODEL_PATH)

metadata = {
    "timestamp": datetime.now().isoformat(),
    "paths": {
        "base_dir": BASE_DIR,
        "model_path": MODEL_PATH,
        "metadata_json": METADATA_JSON,
        "splits": {
            "X_train": X_TRAIN_PATH, "y_train": Y_TRAIN_PATH,
            "X_val": X_VAL_PATH, "y_val": Y_VAL_PATH,
            "X_test": X_TEST_PATH, "y_test": Y_TEST_PATH
        }
    },
    "env": {
        "python": platform.python_version(),
        "platform": platform.platform()
    },
    "data": {
        "n_features": int(X_train.shape[1]),
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(X_test.shape[0]),
        "class_dist": {
            "train": {int(k): int((y_train == k).sum()) for k in [0, 1]},
            "val": {int(k): int((y_val == k).sum()) for k in [0, 1]},
            "test": {int(k): int((y_test == k).sum()) for k in [0, 1]}
        }
    },
    "model": {
        "type": "XGBClassifier",
        "params": {
            "n_estimators": xgb.get_params().get("n_estimators"),
            "max_depth": xgb.get_params().get("max_depth"),
            "learning_rate": xgb.get_params().get("learning_rate"),
            "subsample": xgb.get_params().get("subsample"),
            "colsample_bytree": xgb.get_params().get("colsample_bytree"),
            "reg_lambda": xgb.get_params().get("reg_lambda"),
            "reg_alpha": xgb.get_params().get("reg_alpha"),
            "tree_method": xgb.get_params().get("tree_method"),
            "scale_pos_weight": scale_pos_weight,
            "random_state": SEED,
        },
        "best_iteration": int(getattr(xgb, "best_iteration", -1)),
        "best_threshold": best_threshold
    },
    "metrics_test": {
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc,
        "auc_pr": ap,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
    },
    "training_time_seconds": train_time
}

with open(METADATA_JSON, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f" Save model: {MODEL_PATH}")
print(f" Save metadata: {METADATA_JSON}")

print("\n*"*10,"training completed","*"*10)