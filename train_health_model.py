import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1. Load Dataset
print("Loading health dataset...")
df = pd.read_csv("data.csv")

# Clean column headers
df.columns = df.columns.str.strip()

# 2. Filter & Clean Missing/Suppressed Data
# We focus on predicting 'Age-specific rate (per 100,000)'
df_clean = df[df["measure"] == "Age-specific rate (per 100,000)"].copy()
df_clean = df_clean.dropna(subset=["value"])  # Remove suppressed/missing values

# 3. Define Categorical Features and Target
categorical_cols = ["state", "age_group", "period"]
target_col = "value"

X = df_clean[categorical_cols]
y = df_clean[target_col]

# 4. Data Preprocessing Pipeline (One-Hot Encoding)
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False),
            categorical_cols,
        )
    ]
)

# Convert string categories into binary feature vectors (0s and 1s)
X_processed = preprocessor.fit_transform(X)

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. Build TensorFlow Neural Network Architecture
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),  # Output layer producing continuous rate estimates
    ]
)

# Compile Model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 6. Train the Model
print("Training TensorFlow Neural Network...")
model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
)

# 7. Generate Predictions & Calculate Errors
print("Generating predictions...")
predictions = model.predict(X_processed).flatten()

df_clean["Predicted_Rate"] = predictions
df_clean["Absolute_Error"] = np.abs(df_clean[target_col] - df_clean["Predicted_Rate"])

# 8. Save Processed CSV for Power BI
output_file = "Health_Predictions_PowerBI.csv"
df_clean.to_csv(output_file, index=False)
print(f"\nModel training complete! File saved as '{output_file}'")