import numpy as np
from sklearn.model_selection import train_test_split

#Load features and labels
X = np.load("numpy/root/X_data.npy")
y = np.load("numpy/root/y_labels.npy")

#Split dataset into 70% for train and 30% for validation and test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#Split 30% remain into 15% for validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

#Save train 
np.save("numpy/X_train.npy", X_train)
np.save("numpy/y_train.npy", y_train)

#Save validation
np.save("numpy/X_val.npy", X_val)
np.save("numpy/y_val.npy", y_val)

#Save test
np.save("numpy/X_test.npy", X_test)
np.save("numpy/y_test.npy", y_test)

print("-"*20," OK ","-"*20)