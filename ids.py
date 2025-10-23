import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# load dataset
data = np.load("final_dataset.npy", allow_pickle=True)

# Separate features (X) and labels (y)
X = data[:, :-1]  # select all columns except the last one
y = data[:, -1]   # select the last column, the label

#Split into training/testing sets
test_size = 0.33  # 33% for testing
seed = 7          # ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Create and train model
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate resultss
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
