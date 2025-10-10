import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

csv_path = "C:/Users/mathe/OneDrive/Área de Trabalho/TCC2/IDS/DATASETS/UNSW-NB15/UNSW_NB15_training-set.csv"
df = pd.read_csv(csv_path)

# normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

selected_cols = [
    'id','dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl',
    'sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin',
    'tcprtt','synack','ackdat','smean','dmean','trans_depth','response_body_len','ct_srv_src',
    'ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login',
    'ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports','attack_cat','label'
]

#add columns 
for col in selected_cols:
    if col not in df.columns:
        df[col] = np.nan

df = df[selected_cols]

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('unknown')
    else:
        df[col] = df[col].fillna(0)

#Lable encoding
encoders = {}
for col in ['proto', 'service', 'state', 'attack_cat']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Convert to numpy array
data_array = df.to_numpy(dtype=np.float32)


y = df['label'].to_numpy(dtype=np.int32)
X = df.drop(columns=['label']).to_numpy(dtype=np.float32)

np.save("unsw_nb15_training_set_features.npy", X)
np.save("unsw_nb15_training_set_labels.npy", y)

print("Preprocessing ready!")
print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
