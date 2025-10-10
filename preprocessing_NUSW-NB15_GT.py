import pandas as pd
import numpy as np

csv_path = "/DATASETS/UNSW-NB15/NUSW-NB15_GT.csv"
df = pd.read_csv(csv_path)

# normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

selected_cols = [
    "start_time", "last_time", "attack_category",
    "attack_subcategory", "protocol",
    "source_ip", "source_port",
    "destination_ip", "destination_port"
]

df = df[selected_cols]

# Handle missing values
df.fillna("unknown", inplace=True)

#Lable encoding
from sklearn.preprocessing import LabelEncoder

encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

# Convert to numpy array
data_array = df.to_numpy(dtype=np.float32)


X = df.drop(columns=["attack_category"]).to_numpy(dtype=np.float32)
y = df["attack_category"].to_numpy(dtype=np.int32)

np.save("unsw_nb15_gt_features.npy", X)
np.save("unsw_nb15_gt_labels.npy", y)

print("Preprocessing ready!")
print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
