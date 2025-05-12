import pandas as pd
from io import StringIO

# Load your CSV data
df = pd.read_csv("combined_can_data.csv")  # Replace with your actual file path

# Filter out rows where all Data columns are 0
data_cols = ['Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7']

# Clean and convert data columns to integers, handling hexadecimal
for col in data_cols:
    df[col] = df[col].astype(str).str.lower().str.replace('0x', '', regex=False).apply(lambda x: int(x, 16))

# Filter rows where the sum of data columns is not 0
filtered_df = df[df[data_cols].sum(axis=1) != 0]

# Export the filtered DataFrame to a CSV file
filtered_df.to_csv("filtered_can_data.csv", index=False)  # Saves to 'filtered_can_data.csv'

# Display the filtered DataFrame
print(filtered_df.to_markdown(index=False, numalign="left", stralign="left"))
