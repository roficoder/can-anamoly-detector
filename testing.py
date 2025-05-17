import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import joblib

# === Load test data ===
df = pd.read_csv('new_can_test.csv')  # replace with your test file path

# === Feature Engineering ===
# def feature_engineering(df):
#     df_feat = df.copy()

#     # Convert ID to decimal
#     df_feat['ID_dec'] = df_feat['ID'].apply(lambda x: int(str(x), 16) if isinstance(x, str) else int(x))

#     # Ensure byte columns are integers
#     data_cols = [f'Data{i}' for i in range(8)]
#     df_feat[data_cols] = df_feat[data_cols].fillna(0).astype(int)

#     # Byte-level statistics
#     df_feat['data_sum'] = df_feat[data_cols].sum(axis=1)
#     df_feat['data_mean'] = df_feat[data_cols].mean(axis=1)
#     df_feat['data_std'] = df_feat[data_cols].std(axis=1)
#     df_feat['data_unique_bytes'] = df_feat[data_cols].nunique(axis=1)
#     df_feat['data_max'] = df_feat[data_cols].max(axis=1)
#     df_feat['data_min'] = df_feat[data_cols].min(axis=1)
#     df_feat['data_range'] = df_feat['data_max'] - df_feat['data_min']

#     # Bit counts
#     for col in data_cols:
#         df_feat[f'{col}_bitcount'] = df_feat[col].apply(lambda x: bin(x).count('1'))

#     # Final features
#     selected_features = (
#         data_cols +
#         ['data_unique_bytes', 'data_sum', 'data_mean', 'data_std',
#          'data_max', 'data_min', 'data_range', 'ID_dec'] +
#         [f'{col}_bitcount' for col in data_cols]
#     )
#     return df_feat[selected_features]
def feature_engineering(df):
    df_feat = df.copy()
    data_cols = [f'Data{i}' for i in range(8)]
    df_feat[data_cols] = df_feat[data_cols].fillna(0).astype(int)
    return df_feat[data_cols]


# === Apply feature engineering ===
X_new = feature_engineering(df)

# === Load scaler and transform ===
scaler = joblib.load('saved_models/scaler.save')
X_scaled = scaler.transform(X_new)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# === Load trained models ===
model_cnn = load_model('saved_models/model_cnn.h5')
model_rnn = load_model('saved_models/model_rnn.h5')
model_lstm = load_model('saved_models/model_lstm.h5')
model_att_lstm = load_model('saved_models/model_att_lstm.h5')

# === Predict function ===
def predict_labels(model, X_input):
    preds = model.predict(X_input)
    return np.argmax(preds, axis=1)

# === Make predictions ===
cnn_preds = predict_labels(model_cnn, X_reshaped)
rnn_preds = predict_labels(model_rnn, X_reshaped)
lstm_preds = predict_labels(model_lstm, X_reshaped)
att_lstm_preds = predict_labels(model_att_lstm, X_reshaped)

# === Decode predictions ===
label_encoder = joblib.load('saved_models/label_encoder.save')
df['Pred_CNN'] = label_encoder.inverse_transform(cnn_preds)
df['Pred_RNN'] = label_encoder.inverse_transform(rnn_preds)
df['Pred_LSTM'] = label_encoder.inverse_transform(lstm_preds)
df['Pred_ATT_LSTM'] = label_encoder.inverse_transform(att_lstm_preds)

# === Output results ===
print(df[['Timestamp', 'ID', 'Pred_CNN', 'Pred_RNN', 'Pred_LSTM', 'Pred_ATT_LSTM']])

# === Optional: save to CSV ===
df.to_csv('predicted_can_results.csv', index=False)
