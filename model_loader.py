import pandas as pd
import joblib

# Load the saved model
model = joblib.load("can_model.pkl")

# Example: Load new data (make sure it's processed like before)
new_data = pd.read_csv("testing_data_converted.csv")

# Convert hex to integer for prediction
for col in [f"Data{i}" for i in range(8)]:
    new_data[col] = new_data[col].apply(lambda x: int(str(x), 16) if pd.notnull(x) else 0)

X_new = new_data[[f"Data{i}" for i in range(8)]]

# Make predictions
predictions = model.predict(X_new)

# Output results
new_data['Predicted_Label'] = predictions
print(new_data[['Timestamp', 'ID'] + [f'Data{i}' for i in range(8)] + ['Predicted_Label']])
