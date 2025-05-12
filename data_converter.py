import pandas as pd

# Step 1: Load original dataset
input_file = 'testing_data.csv'          # Replace with your input file name
output_file = 'testing_data_converted.csv'       # Output file with numeric values

df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()      # Remove accidental whitespace in headers

# Step 2: Convert hex columns to numeric
data_columns = [f'Data{i}' for i in range(8)]

for col in data_columns:
    df[col] = df[col].apply(lambda x: int(str(x), 16) if pd.notnull(x) else 0)

# Step 3: Write to new CSV file
df.to_csv(output_file, index=False)

print(f"Hex values converted and saved to '{output_file}'.")
