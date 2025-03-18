# scripts/model_retraining.py
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def main():
    data = np.load("processed_data.npz", allow_pickle=True)
    X = data['X']
    y = data['y']
    
    # Option to load an existing model or create a new one
    try:
        model = load_model("models/model.h5")
        print("Loaded existing model.")
    except Exception as e:
        print("No existing model found, building a new one.")
        model = build_model((X.shape[1], X.shape[2]))
    
    # Split data (time-based or randomized as needed)
    split_idx = int(0.7 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=50, batch_size=32)
    
    model.save("models/model.h5")
    print("Model retraining complete and saved.")

if __name__ == "__main__":
    main()
