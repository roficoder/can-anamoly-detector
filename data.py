import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Load Data ===
file_path = "combined_can_data.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found.")

df = pd.read_csv(file_path)

# === Ensure 'Label' column is present ===
if 'Label' not in df.columns:
    raise KeyError("'Label' column not found in the dataset.")

# === Separate Normal and Abnormd:\car-hacking-data\10) CAN-Intrusion Dataset\label-placer.pyal Data ===
normal_df = df[df['Label'] == 0]
abnormal_df = df[df['Label'] == 1]

# === Compute Mean and Std for Each Column (Excluding 'Label') ===
numeric_cols = df.select_dtypes(include='number').columns.difference(['Label'])
comparison = pd.DataFrame(index=['Normal Mean', 'Abnormal Mean', 'Normal Std', 'Abnormal Std'])

for col in numeric_cols:
    comparison[col] = [
        normal_df[col].mean(),
        abnormal_df[col].mean(),
        normal_df[col].std(),
        abnormal_df[col].std()
    ]

# === Transpose for Better Viewing ===
comparison = comparison.T
print("Summary Statistics Comparison:\n", comparison)

# === Plotting Boxplots ===
# Use 'Label' as 'status' for consistency in plots
df['status'] = df['Label'].map({0: 'Normal', 1: 'Abnormal'})

# Plot for each numeric column
for col in numeric_cols:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x='status', y=col, palette='Set2')
    plt.title(f'Distribution of {col} in Normal vs Abnormal')
    plt.xlabel("Status")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()
