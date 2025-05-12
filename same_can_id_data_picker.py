import pandas as pd

# Load the dataset
df = pd.read_csv("filtered_can_data_attacks.csv")

# Replace 'your_can_id' with the CAN ID you want to filter (e.g., '04b0')
target_can_id = '0260'

# Filter the rows where ID matches the target CAN ID
filtered_df = df[df['ID'] == target_can_id]

# Save the filtered data to a new CSV file
filtered_df.to_csv(f"can_id_{target_can_id}.csv", index=False)

print(f"Filtered data with CAN ID {target_can_id} saved to 'can_id_{target_can_id}.csv'")
