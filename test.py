import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load data
data = pd.read_csv("customer_data.csv")

# Separate features and target
X = data.drop("purchased", axis=1)
y = data["purchased"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)

print("Accuracy:", acc)
print("F1:", f1)
