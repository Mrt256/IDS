import numpy as np
import pandas as pd
import json
import os

FEATURES = ["Destination Port", "Flow Duration", "Total Fwd Packets",
            "Total Backward Packets", "Total Length of Fwd Packets",
            "Total Length of Bwd Packets", "Fwd Packet Length Max",
            "Fwd Packet Length Min", "Fwd Packet Length Mean",
            "Bwd Packet Length Std"]

def preprocess(df: pd.DataFrame) -> np.ndarray: #recive dataframe and return numpy array

    for col in FEATURES: # reorder columns and fill missing with 0
        if col not in df.columns:
            df[col] = 0.0

    df = df.reindex(columns=FEATURES, fill_value=0)
    df = df.fillna(0).astype(np.float32) # treat NaN values and convert to float32
    X = df.to_numpy()

    return X
