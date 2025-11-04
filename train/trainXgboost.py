import os
import json
import time
import numpy as np
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, classification_report
)
from joblib import dump
from datetime import datetime
import platform
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
import pandas as pd

#------------------- FILES -------------------

# Bring path numpy 
BASE_DIR = r"numpy"
OUTPUT_DIR = r"output/xgboost"

#Bring files
X_TRAIN_PATH = os.path.join(BASE_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(BASE_DIR, "y_train.npy")
X_VAL_PATH = os.path.join(BASE_DIR, "X_val.npy")
Y_VAL_PATH = os.path.join(BASE_DIR, "y_val.npy")
X_TEST_PATH = os.path.join(BASE_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(BASE_DIR, "y_test.npy")

#Result files
MODEL_PATH   = os.path.join(OUTPUT_DIR, "model_xgb_ids.pkl")
METADATA_JSON = os.path.join(OUTPUT_DIR, "model_xgb_ids_metadata.json")

SEED = 42
N_SPLITS = 5

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

## -- Training with K-Fold Cross-Validation --

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED) #Divide the data into 5 SPLITS parts

fold_metrics = []

# Loop over each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    print(f"\nFold {fold}/{N_SPLITS}")
    
    # Split data into training and validation sets
    X_tr, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_tr, y_val_fold = y_train[train_idx], y_train[val_idx]

    xgb = XGBClassifier(
        n_estimators=1500, #maximum trees
        max_depth=6, #depth of the trees
        learning_rate=0.08, #boosting learning rate
        subsample=0.8, # sampling
        colsample_bytree=0.8,#sampling
        reg_lambda=2.0, #regularization to prevent overfitting
        reg_alpha=0.0, #regularization to prevent overfitting
        random_state=SEED, # Set a fixed random seed
        n_jobs=-1, #Set the number of cores
        tree_method="hist", #Set the use of cpu instead gpu
        scale_pos_weight=scale_pos_weight, #adjust the weight of the minority class          
        eval_metric="auc", #Area Under the ROC Curve (Receiver Operating Characteristic)
        importance_type="gain" #Feature importance based on the average gain
    )

    # Train the model and measure training time
    t0 = time.time()
    xgb.fit(X_tr, y_tr, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
    train_time = time.time() - t0

    # Predictions and probabilities on folder
    y_pred_fold = xgb.predict(X_val_fold)
    y_prob_fold = xgb.predict_proba(X_val_fold)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y_val_fold, y_pred_fold)
    pre = precision_score(y_val_fold, y_pred_fold, zero_division=0)
    rec = recall_score(y_val_fold, y_pred_fold, zero_division=0)
    f1 = f1_score(y_val_fold, y_pred_fold, zero_division=0)
    auc = roc_auc_score(y_val_fold, y_prob_fold)

    # Store metrics for the fold
    fold_metrics.append({
        "fold": fold, "acc": acc, "prec": pre, "rec": rec, "f1": f1, "auc": auc, "train_time": train_time
    })

    print(f"Fold {fold} — Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Time={train_time:.1f}s")

# Show training time for the last fold
train_time = time.time() - t0
print(f"Train time: {train_time:.1f}s")

# Show used trees
best_iter = getattr(xgb, "best_iteration", None)
if best_iter is not None:
    print(f"Used trees (best_iteration): {best_iter + 1}\n")
else:
    print(f"Used trees: {xgb.get_params().get('n_estimators')} (no early stopping)\n")

##----------- Final training model -----------
final_xgb = XGBClassifier(
    n_estimators=1500,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=0.0,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr" # optimized for Area Under the Precision-Recall Curve
)

# Train the final model on the entire training set
t0 = time.time()
final_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
final_train_time = time.time() - t0
print(f"\nFinal model train time: {final_train_time:.1f}s")

#--------- Choose the best threshold ---------

val_prob = final_xgb.predict_proba(X_val)[:, 1]

#Sweep possible thresholds from the Precision-Recall curve
prec, rec, thr = precision_recall_curve(y_val, val_prob)

#Avoids division by zero and selects the threshold with the highest F1 score
f1_scores = (2 * prec * rec) / np.clip(prec + rec, a_min=1e-12, a_max=None)
best_idx = int(np.argmax(f1_scores))
best_threshold = 0.39 if best_idx >= len(thr) else float(thr[best_idx])

print(f"Best threshold (validation, max F1): {best_threshold:.6f}")
print(f"F1(val)={f1_scores[best_idx]:.4f} | Prec(val)={prec[best_idx]:.4f} | Rec(val)={rec[best_idx]:.4f}\n")

# --------- Evaluate on the test set ---------

test_prob = final_xgb.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

#Calculate main metrics
acc = accuracy_score(y_test, test_pred)
pre = precision_score(y_test, test_pred, zero_division=0)
rec = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
auc = roc_auc_score(y_test, test_prob)
ap = average_precision_score(y_test, test_prob)  # AUC-PR

#Confusion Matrix and detailed values
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

# ---------- K-Fold Summary ----------
print("\nK-Fold:")
mean_acc = np.mean([m["acc"] for m in fold_metrics])
mean_f1 = np.mean([m["f1"] for m in fold_metrics])
mean_auc = np.mean([m["auc"] for m in fold_metrics])
print(f"average accuracy: {mean_acc:.4f}")
print(f"average F1-score: {mean_f1:.4f}")
print(f"average AUC: {mean_auc:.4f}")

# ---------- Feature Importance ----------
importance = final_xgb.feature_importances_
feat_names = ["Destination Port", "Flow Duration", "Total Fwd Packets",
              "Total Backward Packets",	"Total Length of Fwd Packets",	
              "Total Length of Bwd Packets", "Fwd Packet Length Max",	
              "Fwd Packet Length Min", "Fwd Packet Length Mean", "Bwd Packet Length Std"]
imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importance}).sort_values(by="Importance", ascending=False)

imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

#---------- Save model and metadata ----------

# Save the trained model
dump(final_xgb, MODEL_PATH)

# Metadata dictionary
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
    "env": { # environment details
        "xgboost_version": xgboost.__version__,
        "xgboost_params": final_xgb.get_params(),
        "sklearn_version": pd.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine()
    },
    "data": { # dataset details
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
    "model": { # model details
        "type": "XGBClassifier",
    "params": {
        "n_estimators": final_xgb.get_params().get("n_estimators"),
        "max_depth": final_xgb.get_params().get("max_depth"),
        "learning_rate": final_xgb.get_params().get("learning_rate"),
        "subsample": final_xgb.get_params().get("subsample"),
        "colsample_bytree": final_xgb.get_params().get("colsample_bytree"),
        "reg_lambda": final_xgb.get_params().get("reg_lambda"),
        "reg_alpha": final_xgb.get_params().get("reg_alpha"),
        "tree_method": final_xgb.get_params().get("tree_method"),
        "scale_pos_weight": scale_pos_weight,
        "random_state": SEED,
    },
        "best_iteration": int(getattr(final_xgb, "best_iteration", -1)),
        "best_threshold": best_threshold
    },
    "metrics_test": { # final test metrics
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
    "training_time_seconds": final_train_time
}

# Add K-Fold results to metadata
metadata["kfold_results"] = fold_metrics
metadata["kfold_mean"] = {
    "accuracy": float(mean_acc),
    "f1": float(mean_f1),
    "auc": float(mean_auc)
}

# Save metadata to JSON file
with open(METADATA_JSON, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)


# Final messages
print(f" Save model: {MODEL_PATH}")
print(f" Save metadata: {METADATA_JSON}")

print("\n","*"*10,"training completed","*"*10)