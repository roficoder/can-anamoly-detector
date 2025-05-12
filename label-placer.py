import pandas as pd

def parse_can_line(line, label_name):
    # Binary label map: 0 for Normal, 1 for all abnormal types
    label_map = {
        'Normal': 0,
        'Fuzzy': 1,
        'DoS': 1,
        'Impersonation': 1
    }
    try:
        parts = line.split()
        timestamp = float(parts[1])
        can_id = parts[3]
        dlc = int(parts[6])
        data = parts[8:8+8]
        if len(data) < 8:
            data += ['00'] * (8 - len(data))  # Pad if less than 8 bytes
        return {
            'Timestamp': timestamp,
            'ID': can_id,
            'DLC': dlc,
            'Data0': data[0],
            'Data1': data[1],
            'Data2': data[2],
            'Data3': data[3],
            'Data4': data[4],
            'Data5': data[5],
            'Data6': data[6],
            'Data7': data[7],
            'Label': label_map[label_name]
        }
    except Exception:
        return None

def process_log_file(filename, label, limit=None):
    data = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            parsed = parse_can_line(line.strip(), label)
            if parsed:
                data.append(parsed)
    return data

# === File Inputs ===
normal_logs = process_log_file('normal.txt', 'Normal')
dos_logs = process_log_file('dos-attack.txt', 'DoS')
fuzzy_logs = process_log_file('fuzzy.txt', 'Fuzzy')
imp_logs = process_log_file('impersonation.txt', 'Impersonation')

# === Combine and Save ===
all_logs = normal_logs + dos_logs + fuzzy_logs + imp_logs
df = pd.DataFrame(all_logs)
df.to_csv('combined_can_data_attacks.csv', index=False)

print("✅ Data successfully saved to combined_can_data.csv")
