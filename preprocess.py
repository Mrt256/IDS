import numpy as np
import pandas as pd
import json
import os

MODEL_METADATA_PATH = os.path.join("output", "xgboost", "model_xgb_ids_metadata.json") #read the metadata file

def preprocess(df: pd.DataFrame) -> np.ndarray: #recive dataframe and return numpy array
    try:
        with open(MODEL_METADATA_PATH, "r") as f:
            metadata = json.load(f)
            FEATURES = metadata["features"]
    except Exception as e:
        raise FileNotFoundError(f"ERROR: files not found: {e}")

    for col in FEATURES: # reorder columns and fill missing with 0
        if col not in df.columns:
            df[col] = 0.0

    df = df.reindex(columns=FEATURES, fill_value=0)
    df = df.fillna(0).astype(np.float32) # treat NaN values and convert to float32
    X = df.to_numpy()

    return X
