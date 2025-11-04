import os
import sys
import csv
import json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FALSE_POS_PATH = os.path.join(BASE_DIR, "false_positives.csv")

FEATURE_NAMES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Std"
]

FIELDNAMES = FEATURE_NAMES + ["Label", "note"]

def save_false_positive(flow_data):

    os.makedirs(os.path.dirname(FALSE_POS_PATH), exist_ok=True)
    header_needed = not os.path.exists(FALSE_POS_PATH)

    record = {k: flow_data.get(k, 0) for k in FEATURE_NAMES}
    record["Label"] = 0
    record["note"] = "False Positive"

    with open(FALSE_POS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if header_needed:
            writer.writeheader()
        writer.writerow(record)

if __name__ == "__main__":
    json_path = sys.argv[1]
    with open(json_path, "r") as f:
        flow_data = json.load(f)
    save_false_positive(flow_data)
