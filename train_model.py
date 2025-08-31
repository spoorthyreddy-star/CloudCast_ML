# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample enhanced dataset with added feature placeholders
data = [
    # [Temp, Humidity, Pressure, Wind, Clouds, Sunrise, Sunset, DescriptionCat, Rain]
    [28.0, 88, 1002, 3.4, 90, 1625467200, 1625517600, 1, 1],  # Rain
    [26.5, 91, 1003, 2.8, 95, 1625467300, 1625517700, 1, 1],
    [29.0, 85, 1001, 3.9, 85, 1625467400, 1625517800, 1, 1],
    [22.0, 60, 1015, 1.5, 10, 1625467100, 1625517500, 0, 0],  # Clear
    [30.1, 45, 1011, 2.0, 5, 1625467000, 1625517400, 0, 0],
    [25.5, 75, 1013, 2.6, 20, 1625466900, 1625517300, 0, 0],
    [27.3, 68, 1008, 3.0, 30, 1625466800, 1625517200, 0, 0],
]

columns = [
    "Temperature", "Humidity", "Pressure", "Wind Speed", "Clouds",
    "Sunrise", "Sunset", "DescriptionCategory", "RainTomorrow"
]

# Build DataFrame
df = pd.DataFrame(data, columns=columns)

# Features and target
X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "raincast_model.pkl")
print("✅ Model trained and saved as raincast_model.pkl")
