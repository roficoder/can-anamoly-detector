import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("filtered_can_data.csv")

# ========== Convert Hex Data (if needed) ==========
# If data bytes are in hex (which seems already processed), you can skip this step
# Otherwise uncomment below:
# for col in ['Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7']:
#     df[col] = df[col].apply(lambda x: int(str(x), 16))

# ========== Basic Feature Engineering ==========

# 1. Message Entropy: How many unique values per row (useful for detecting randomization)
df["data_unique_bytes"] = df[['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']].nunique(axis=1)

# 2. Byte Sum & Mean — total signal strength
df["data_sum"] = df[['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']].sum(axis=1)
df["data_mean"] = df[['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']].mean(axis=1)

# 3. Byte Standard Deviation — randomness or noise level
df["data_std"] = df[['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']].std(axis=1)

# 4. Pairwise Differences — useful for detecting jumps in bytes
df["data_max"] = df[['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']].max(axis=1)
df["data_min"] = df[['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']].min(axis=1)
df["data_range"] = df["data_max"] - df["data_min"]

# 5. Temporal Feature — Time since last frame (can help detect flood attacks)
df["time_diff"] = df["Timestamp"].diff().fillna(0)

# 6. Frequency Feature — Rolling count of same ID in last N messages
df["ID_count_10"] = df.groupby("ID")["ID"].transform(lambda x: x.rolling(window=10, min_periods=1).count())

# 7. Bit-Level Features — Count 1s in each byte (for entropy-like behavior)
for col in ['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']:
    df[f'{col}_bitcount'] = df[col].apply(lambda x: bin(int(x)).count('1'))

# 8. ID-based features — Convert ID from hex to int if needed
df['ID_dec'] = df['ID'].apply(lambda x: int(str(x), 16) if isinstance(x, str) else int(x))

# 9. Normalized byte positions — each byte relative to sum
for col in ['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']:
    df[f'{col}_norm'] = df[col] / df["data_sum"].replace(0, 1)

# ========== Feature Set Ready ==========

# Drop unused columns if needed
# df.drop(columns=["Timestamp", "ID"], inplace=True)

# Save or return the new dataframe
df.to_csv("engineered_can_data.csv", index=False)
