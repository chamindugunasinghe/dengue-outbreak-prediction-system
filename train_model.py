import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading dengue data...")
df = pd.read_csv('data/dengue_data.csv')

# Prepare features
features = ['District', 'Month', 'Week', 'Avg Max Temp (°C)', 'Avg Min Temp (°C)', 
            'Total Precipitation (mm)', 'Avg Wind Speed (km/h)']

# Encode District
le = LabelEncoder()
df['District_Encoded'] = le.fit_transform(df['District'])

# Create risk levels based on cases
# Low: 0-3, Medium: 4-9, High: 10+
df['Risk'] = pd.cut(df['Number_of_Cases'], bins=[-1, 3, 9, float('inf')], labels=[0, 1, 2]).astype(int)

# Select features and target
feature_cols = ['District_Encoded', 'Month', 'Week', 'Avg Max Temp (°C)', 'Avg Min Temp (°C)', 
                'Total Precipitation (mm)', 'Avg Wind Speed (km/h)']
X = df[feature_cols]
y = df['Risk']

# Remove rows with missing values
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

print(f"Training data shape: {X.shape}")
print(f"Risk distribution:\n{y.value_counts().sort_index()}")

# Train model
print("\nTraining XGBoost model...")
model = XGBClassifier(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
model.fit(X, y)

# Save trained model
joblib.dump(model, 'models/dengue_model.pkl')
print("✓ Model trained and saved to models/dengue_model.pkl")

# Test it
print("\nTesting model...")
test_pred = model.predict(X[:5])
print(f"Sample predictions: {test_pred}")
print("✓ Model is working correctly!")
