import os
import json
import time
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, classification_report
)
from joblib import dump
from datetime import datetime
import platform
from sklearn.model_selection import StratifiedKFold
import pandas as pd

#------------------- FILES -------------------

BASE_DIR = r"numpy"
OUTPUT_DIR = r"output/lightgbm"

# Bring files
X_TRAIN_PATH = os.path.join(BASE_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(BASE_DIR, "y_train.npy")
X_VAL_PATH = os.path.join(BASE_DIR, "X_val.npy")
Y_VAL_PATH = os.path.join(BASE_DIR, "y_val.npy")
X_TEST_PATH = os.path.join(BASE_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(BASE_DIR, "y_test.npy")

# Result files
MODEL_PATH   = os.path.join(OUTPUT_DIR, "model_lgb_ids.pkl")
METADATA_JSON = os.path.join(OUTPUT_DIR, "model_lgb_ids_metadata.json")

SEED = 42
N_SPLITS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
X_train = np.load(X_TRAIN_PATH).astype(np.float32)
y_train = np.load(Y_TRAIN_PATH).astype(np.int32)
X_val = np.load(X_VAL_PATH).astype(np.float32)
y_val = np.load(Y_VAL_PATH).astype(np.int32)
X_test = np.load(X_TEST_PATH).astype(np.float32)
y_test = np.load(Y_TEST_PATH).astype(np.int32)

#------------- Print Distribution -------------

def distribution_sum(y, name):
    uniq, cnt = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
    total = int(y.shape[0])
    print(f" {name}: {dist} (total={total:,})")

distribution_sum(y_train, "Train")
distribution_sum(y_val, "Validation")
distribution_sum(y_test, "Test")
print("/n")

#-----------  Imbalance Adjustment -----------

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = float(n_neg) / max(float(n_pos), 1.0)
print(f"scale_pos_weight: {scale_pos_weight:.3f}  (neg={n_neg:,} / pos={n_pos:,})\n")

#--------- Define and train the model ---------

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    print(f"\nFold {fold}/{N_SPLITS}")
    X_tr, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_tr, y_val_fold = y_train[train_idx], y_train[val_idx]

    params = {
        "n_estimators": 1500,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "reg_alpha": 0.0,
        "random_state": SEED,
        "n_jobs": -1,
        "objective": "binary",
        "boosting_type": "gbdt",
        "scale_pos_weight": scale_pos_weight,
        "metric": "auc",
        "verbosity": -1 
    }

    model = lgb.LGBMClassifier(**params)

    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_val_fold, y_val_fold)])
    train_time = time.time() - t0

    y_pred_fold = model.predict(X_val_fold)
    y_prob_fold = model.predict_proba(X_val_fold)[:, 1]

    acc = accuracy_score(y_val_fold, y_pred_fold)
    pre = precision_score(y_val_fold, y_pred_fold, zero_division=0)
    rec = recall_score(y_val_fold, y_pred_fold, zero_division=0)
    f1 = f1_score(y_val_fold, y_pred_fold, zero_division=0)
    auc = roc_auc_score(y_val_fold, y_prob_fold)

    fold_metrics.append({
        "fold": fold, "acc": acc, "prec": pre, "rec": rec, "f1": f1, "auc": auc, "train_time": train_time
    })

    print(f"Fold {fold} — Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Time={train_time:.1f}s")

print("\nTraining completed across all folds.\n")

#----------- Final training model -----------

final_model = lgb.LGBMClassifier(**params)

t0 = time.time()
final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
final_train_time = time.time() - t0
print(f"\nFinal model train time: {final_train_time:.1f}s")

#--------- Choose the best threshold ---------

val_prob = final_model.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, val_prob)
f1_scores = (2 * prec * rec) / np.clip(prec + rec, a_min=1e-12, a_max=None)
best_idx = int(np.argmax(f1_scores))
best_threshold = 0.39 if best_idx >= len(thr) else float(thr[best_idx])

print(f"Best threshold (validation, max F1): {best_threshold:.6f}")
print(f"F1(val)={f1_scores[best_idx]:.4f} | Prec(val)={prec[best_idx]:.4f} | Rec(val)={rec[best_idx]:.4f}\n")

# --------- Evaluate on the test set ---------

test_prob = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

acc = accuracy_score(y_test, test_pred)
pre = precision_score(y_test, test_pred, zero_division=0)
rec = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
auc = roc_auc_score(y_test, test_prob)
ap = average_precision_score(y_test, test_prob)
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
mean_acc = np.mean([m["acc"] for m in fold_metrics])
mean_f1 = np.mean([m["f1"] for m in fold_metrics])
mean_auc = np.mean([m["auc"] for m in fold_metrics])

print("\nK-Fold Summary:")
print(f"Average Accuracy: {mean_acc:.4f}")
print(f"Average F1-score: {mean_f1:.4f}")
print(f"Average AUC: {mean_auc:.4f}")

# ---------- Feature Importance ----------
importance = final_model.feature_importances_
feat_names = ["Destination Port", "Flow Duration", "Total Fwd Packets",
              "Total Backward Packets", "Total Length of Fwd Packets",
              "Total Length of Bwd Packets", "Fwd Packet Length Max",
              "Fwd Packet Length Min", "Fwd Packet Length Mean", "Bwd Packet Length Std"]

imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importance}).sort_values(by="Importance", ascending=False)
imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

#---------- Save model and metadata ----------
dump(final_model, MODEL_PATH)

metadata = {
    "timestamp": datetime.now().isoformat(),
    "model_type": "LightGBM",
    "params": params,
    "best_threshold": best_threshold,
    "metrics_test": {
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc,
        "auc_pr": ap,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    },
    "kfold_mean": {"accuracy": mean_acc, "f1": mean_f1, "auc": mean_auc},
    "training_time_seconds": final_train_time
}

with open(METADATA_JSON, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\nSave model: {MODEL_PATH}")
print(f"Save metadata: {METADATA_JSON}")
print("\n********** training completed **********")
