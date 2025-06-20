import os
import numpy as np
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

DATASET_PATH = "counting dataset"
LABELS = ['eight', 'nine', 'six', 'two', 'zero', 'unknown']
SAMPLES_PER_FILE = 16000
N_MELS = 64


def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.normalize(y)
    if len(y) > SAMPLES_PER_FILE:
        y = y[:SAMPLES_PER_FILE]
    else:
        y = np.pad(y, (0, SAMPLES_PER_FILE - len(y)), 'constant')

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db


def load_dataset():
    X, y = [] , []
    for label_idx, label in enumerate(LABELS):
        folder = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                path = os.path.join(folder, file)
                mel_db = preprocess_audio(path)
                X.append(mel_db)
                y.append(label_idx)
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y


# Load data
X, y = load_dataset()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle imbalance
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(class_weights))

# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X.shape[1:]),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(LABELS))  # Raw logits
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          batch_size=16,
          epochs=10,
          class_weight=class_weights)

model.save("audio_model_with_unknown.keras")

