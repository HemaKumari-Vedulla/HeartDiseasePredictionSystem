# ====== Heart Disease Prediction (Using 3 Features) ======

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
data = pd.read_csv("heart.csv")

# ----- Encode ChestPainType ONLY and keep encoder -----
cp_encoder = LabelEncoder()
data["ChestPainType"] = cp_encoder.fit_transform(data["ChestPainType"])

# Encode OTHER categorical columns (if needed)
for col in data.columns:
    if data[col].dtype == "object" and col != "ChestPainType":  
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# ===== Use ONLY required 3 features =====
X = data[["Cholesterol", "RestingBP", "ChestPainType"]]
y = data["HeartDisease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===== User Input Prediction =====
print("Enter the following details:")

chol = int(input("Cholesterol: "))
restbp = int(input("Resting Blood Pressure: "))
cp_text = input("Chest Pain Type (ATA/NAP/ASY/TA): ")

# Convert user chest pain input to number using SAME encoder
cp_value = cp_encoder.transform([cp_text])[0]

# Arrange input
user_input = pd.DataFrame([[chol, restbp, cp_value]],
                          columns=["Cholesterol", "RestingBP", "ChestPainType"])
#Scale input
user_input_scaled = scaler.transform(user_input)


# Predict
prediction = model.predict(user_input_scaled)

# Output
if prediction[0] == 1:
    print("\nThe person is likely to have Heart Disease.")
else:
    print("\nThe person is NOT likely to have Heart Disease.")
