import os
import json
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Masking
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

# Define paths (adjust as necessary)
PROCESSED_DATA_PATH = os.path.abspath("../data/processed")
STATIC_FEATURE_NAMES_FILE = os.path.join(PROCESSED_DATA_PATH, "static_feature_names.json")

# Load the actual static feature names from preprocessing
with open(STATIC_FEATURE_NAMES_FILE, "r") as f:
    static_feature_names = json.load(f)

# List all .npz files in the processed folder
all_npz_files = [os.path.join(PROCESSED_DATA_PATH, f) for f in os.listdir(PROCESSED_DATA_PATH) if f.endswith('.npz')]
if not all_npz_files:
    raise FileNotFoundError(f"No .npz files found in {PROCESSED_DATA_PATH}")

# Step 1: Determine the majority shape among the datasets
shape_to_files = {}
for file in all_npz_files:
    data = np.load(file)
    X = data["X"]
    shape = X.shape  # Shape: (num_sequences, window_size, num_features)
    shape_to_files.setdefault(shape, []).append(file)

# Determine the majority shape by counting occurrences
all_shapes = [shape for shape, files in shape_to_files.items() for _ in range(len(files))]
majority_shape = Counter(all_shapes).most_common(1)[0][0]
print("Majority shape (num_sequences, window_size, num_features):", majority_shape)

# Step 2: Filter the npz files to include only those matching the majority shape
valid_npz_files = shape_to_files[majority_shape]
print("\nFiles with the majority shape:")
for file in valid_npz_files:
    print(file)

# Step 3: Define a generator that yields individual sequences and targets only from the valid files
def data_generator():
    for file in valid_npz_files:
        data = np.load(file)
        X = data["X"]  # Shape: (num_sequences, window_size, num_features)
        y = data["y"]  # Shape: (num_sequences,)
        for i in range(X.shape[0]):
            yield X[i], y[i]

# Determine input shape from a sample file (use valid_npz_files)
sample_data = np.load(valid_npz_files[0])
input_shape = sample_data["X"].shape[1:]  # (window_size, num_features)
print("\nInput shape for the model:", input_shape)

# Step 4: Create a tf.data.Dataset from the generator
dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=input_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
)

# Shuffle, batch, and prefetch the dataset for efficiency
batch_size = 32
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Step 5: Build the LSTM model
model = Sequential([
    Input(shape=input_shape),
    # Optional: Masking layer in case you pad sequences with zeros
    Masking(mask_value=0.0),
    # LSTM layer to process the time-series data
    LSTM(64, return_sequences=False),
    # Final dense layer for regression (predicting a single value)
    Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# Step 6: Define callbacks for training
def lr_schedule(epoch, lr):
    # Example: decay learning rate every 5 epochs
    if epoch % 5 == 0 and epoch:
        return lr * 0.5
    return lr

callbacks = [
    ModelCheckpoint("../model/best_model.keras", save_best_only=True, monitor="loss", mode="min"),
    LearningRateScheduler(lr_schedule, verbose=1)
]

# Train the model
num_epochs = 10
history = model.fit(dataset, epochs=num_epochs, callbacks=callbacks)
