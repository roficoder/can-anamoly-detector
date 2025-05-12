import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('converted_data.csv')  # Replace with your actual filename

# Convert column names if needed (e.g., stripping whitespace)
df.columns = df.columns.str.strip()

# Ensure 'Label' is of integer type
df['Label'] = df['Label'].astype(int)

# Split into normal and abnormal
normal_df = df[df['Label'] == 0]
abnormal_df = df[df['Label'] == 1]

# Data columns to plot
data_columns = [f'Data{i}' for i in range(8)]

# Plot KDE for each data column
for col in data_columns:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(normal_df[col], label='Normal (Label=0)', fill=True, color='blue')
    if not abnormal_df.empty:
        sns.kdeplot(abnormal_df[col], label='Abnormal (Label=1)', fill=True, color='red')
    plt.title(f'KDE Plot for {col}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()
