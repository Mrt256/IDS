import os
import csv
from joblib import load
import numpy as np
from winotify import Notification, audio


LOG_PATH = "logs/ids_log.csv"
MODEL_PATH = "output/xgboost/model_xgb_ids.pkl"
script_path = os.path.abspath("logs/fp_register.py")
temp_path = os.path.abspath("logs/temp.json")

model = load(MODEL_PATH)
IMPORTANCES = model.feature_importances_

FIELDNAMES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Label"
]

def notify_attack(port, reasons):
    
    toast = Notification(
        app_id="IDS",
        title="ATTACK",
        msg=f"Port: {port}\nReasons: {', '.join(reasons)}",
        duration="short"
    )
    toast.set_audio(audio.Default, loop=False)

    toast.add_actions(label="False Positive",launch=f"python \"{script_path}\" \"{temp_path}\"")

    toast.show()

def log_result(result):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    header_needed = not os.path.exists(LOG_PATH)

    filtered = {k: result.get(k, 0) for k in FIELDNAMES}

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if header_needed:
            writer.writeheader()
        writer.writerow(filtered)

    if filtered["Label"] == 1:
        top_idx = np.argsort(IMPORTANCES)[-3:][::-1]
        reasons = [FIELDNAMES[i] for i in top_idx]
        notify_attack(filtered["Destination Port"], reasons)

    else:
        pass