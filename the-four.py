import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense, SimpleRNN,
                                     LSTM, Input, Permute, Multiply, Activation,
                                     RepeatVector, Lambda)
import tensorflow.keras.backend as K
import os

# === Load and preprocess data ===
df = pd.read_csv('filtered_can_data.csv')

# Convert hex data fields to integers
for col in ['Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7']:
    df[col] = df[col].apply(lambda x: int(str(x), 16))

X = df[['Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7']]
y = df['Label']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for time-series models
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Create and Train Models ===

# Directory to save models
os.makedirs("saved_models", exist_ok=True)

# === CNN Model ===
model_cnn = Sequential([
    Conv1D(32, kernel_size=1, activation='relu', input_shape=(1, 8)),
    MaxPooling1D(pool_size=1),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))
model_cnn.save('saved_models/model_cnn.h5')

# === RNN Model ===
model_rnn = Sequential([
    SimpleRNN(64, input_shape=(1, 8)),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])
model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_rnn.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))
model_rnn.save('saved_models/model_rnn.h5')

# === LSTM Model ===
model_lstm = Sequential([
    LSTM(64, input_shape=(1, 8)),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])
model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))
model_lstm.save('saved_models/model_lstm.h5')

# === Attention-based LSTM Model ===
def attention_layer(inputs):
    attention = Dense(1, activation='tanh')(inputs)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(inputs.shape[-1])(attention)
    attention = Permute([2, 1])(attention)
    output_attention = Multiply()([inputs, attention])
    return Lambda(lambda x: K.sum(x, axis=1))(output_attention)

inputs = Input(shape=(1, 8))
lstm_out = LSTM(64, return_sequences=True)(inputs)
att_out = attention_layer(lstm_out)
dense = Dense(64, activation='relu')(att_out)
output = Dense(y_categorical.shape[1], activation='softmax')(dense)
model_att_lstm = Model(inputs=inputs, outputs=output)
model_att_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_att_lstm.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))
model_att_lstm.save('saved_models/model_att_lstm.h5')

# === Evaluation ===
def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')

    return accuracy, precision, f1

# === Load models later (when needed) ===
model_cnn_loaded = load_model('saved_models/model_cnn.h5')
model_rnn_loaded = load_model('saved_models/model_rnn.h5')
model_lstm_loaded = load_model('saved_models/model_lstm.h5')
model_att_lstm_loaded = load_model('saved_models/model_att_lstm.h5')

# === Metrics Table ===
metrics = {
    'Model': ['CNN', 'RNN', 'LSTM', 'Attention-based LSTM'],
    'Accuracy': [
        calculate_metrics(model_cnn_loaded, X_test, y_test)[0],
        calculate_metrics(model_rnn_loaded, X_test, y_test)[0],
        calculate_metrics(model_lstm_loaded, X_test, y_test)[0],
        calculate_metrics(model_att_lstm_loaded, X_test, y_test)[0]
    ],
    'Precision': [
        calculate_metrics(model_cnn_loaded, X_test, y_test)[1],
        calculate_metrics(model_rnn_loaded, X_test, y_test)[1],
        calculate_metrics(model_lstm_loaded, X_test, y_test)[1],
        calculate_metrics(model_att_lstm_loaded, X_test, y_test)[1]
    ],
    'F1 Score': [
        calculate_metrics(model_cnn_loaded, X_test, y_test)[2],
        calculate_metrics(model_rnn_loaded, X_test, y_test)[2],
        calculate_metrics(model_lstm_loaded, X_test, y_test)[2],
        calculate_metrics(model_att_lstm_loaded, X_test, y_test)[2]
    ]
}

metrics_df = pd.DataFrame(metrics)
print(metrics_df)
