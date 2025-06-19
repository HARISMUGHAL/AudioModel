# 🔊 Spoken Digit Classification API (Flask + TensorFlow)

This project is an **Audio Classification System** that identifies spoken digits (`zero`, `two`, `six`, `eight`, `nine`) using a **Convolutional Neural Network (CNN)** trained on a custom audio dataset. The model is deployed using **Flask** as a REST API.

---

## 📁 Dataset

- Dataset: Custom spoken digits dataset (downloaded from Kaggle)
- Folder structure expected:
  ```
  counting dataset/
  ├── zero/
  │   ├── file1.wav
  │   └── ...
  ├── two/
  ├── six/
  ├── eight/
  └── nine/
  ```

---

## 📊 Model Architecture

- Input: Mel-spectrogram of audio (64 Mel bands)
- Layers:
  - 2 × Conv2D + MaxPooling2D + BatchNorm
  - Flatten + Dense + Dropout
  - Output layer with logits for 5 classes
- Framework: **TensorFlow / Keras**

---

## 🔁 Preprocessing

- Sample Rate: 16,000 Hz
- Duration: Padded or trimmed to 1 second (16000 samples)
- Features: **Mel-Spectrogram**
- Normalization: Per audio waveform
- Labels: Encoded as integers (0 to 4) corresponding to:
  - `['eight', 'nine', 'six', 'two', 'zero']`

---

## 🧠 Training Details

- Loss: `SparseCategoricalCrossentropy(from_logits=True)`
- Optimizer: `Adam`
- Epochs: 10
- Batch size: 16
- Model saved as: `audio_model4.keras`

---

## 🚀 API Endpoints (Flask)

### 1. `GET /`

Returns welcome message and expected usage.

### 2. `POST /predict`

- Accepts: Audio file (`.wav`, `.mp3`, `.ogg`, `.flac`) via `form-data` (key: `file`)
- Returns:
  - Predicted label
  - Confidence score
  - All class probabilities
- Rejects:
  - Low confidence predictions (< 0.8 confidence)

Example Request using `curl`:
```bash
curl -X POST http://localhost:5000/predict   -F "file=@sample.wav"
```

---

## 🛠️ Installation & Run Instructions

### 🔹 Step 1: Install Python dependencies

```bash
pip install tensorflow flask librosa numpy scikit-learn
```

### 🔹 Step 2: Train the model (Optional if `audio_model4.keras` already exists)

```bash
python train_model.py
```

### 🔹 Step 3: Run the Flask server

```bash
python app.py
```

---

## 🧪 Example Output (JSON)

```json
{
  "predicted_class": "six",
  "confidence": 0.9231,
  "all_predictions": {
    "eight": 0.0032,
    "nine": 0.0151,
    "six": 0.9231,
    "two": 0.0340,
    "zero": 0.0246
  }
}
```

---

## 📦 Files in This Project

| File | Description |
|------|-------------|
| `train_model.py` | Script to load dataset, preprocess audio, and train CNN model |
| `app.py`         | Flask server for model inference via HTTP |
| `audio_model4.keras` | Trained Keras model |
| `counting dataset/` | Dataset with audio files for digits |

---

## 👨‍💻 Author

Developed as part of an audio ML classification project using TensorFlow and Flask.

---

## 📌 Notes

- Ensure the audio file is in the correct format and length.
- You can add more classes and retrain the model easily.
- Dataset quality highly influences model accuracy.