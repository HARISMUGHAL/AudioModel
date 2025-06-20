
# 🔊 Spoken Digit Classification API (Flask + TensorFlow)

This project is an **Audio Classification System** that identifies spoken digits (`zero`, `two`, `six`, `eight`, `nine`) and can also detect **unknown words** using a **CNN model** trained on Mel-spectrograms. The model is served using a **Flask API** for real-time predictions.

---

## 📁 Dataset Structure

```
counting dataset/
├── zero/
├── two/
├── six/
├── eight/
├── nine/
└── unknown/
     ├── five.wav
     ├── stop.wav
     └── ...
```

- Each folder must contain `.wav` audio files for its respective class.
- `unknown/` contains unrelated spoken words like "five", "stop", etc., used to teach the model what is **not** a digit.

---

## 📊 Model Overview

- **Input**: Mel-spectrogram (64 Mel bands, padded to 1 second)
- **Model**: CNN (Convolutional Neural Network)
- **Layers**:
  - 2 × Conv2D → MaxPooling2D → BatchNorm
  - Flatten → Dense → Dropout → Output logits
- **Output**: Probabilities for 6 classes (`five digits + unknown`)
- **Framework**: TensorFlow / Keras

---

## 🧪 Training Details

- **Preprocessing**:
  - Sample rate: 16,000 Hz
  - Duration: 1 second (padded/truncated)
  - Feature: Mel-spectrogram (64 bands)
- **Label Encoding**:
  ```python
  label_names = ['eight', 'nine', 'six', 'two', 'zero', 'unknown']
  ```
- **Loss**: `SparseCategoricalCrossentropy(from_logits=True)`
- **Optimizer**: `Adam`
- **Epochs**: 10
- **Batch size**: 16
- **Saved model**: `audio_model4.keras`

---

## 🚀 Flask API Endpoints

### ✅ `GET /`

Returns usage instructions and metadata.

### 🎤 `POST /predict`

- Accepts: `.wav`, `.mp3`, `.ogg`, `.flac` via `form-data` (key = `file`)
- Returns:
  - `predicted_class`
  - `confidence` score
  - `all_predictions` (probability for each class)
- If confidence is below 0.8, returns:
  ```json
  {
    "error": "Low confidence prediction",
    "predicted_class": "unknown",
    "confidence": 0.57
  }
  ```

### Example `curl` Request

```bash
curl -X POST http://localhost:5000/predict   -F "file=@example.wav"
```

---

## 🧪 Sample API Output

```json
{
  "predicted_class": "unknown",
  "confidence": 0.77,
  "all_predictions": {
    "eight": 0.002,
    "nine": 0.001,
    "six": 0.003,
    "two": 0.005,
    "zero": 0.007,
    "unknown": 0.982
  }
}
```

---

## 🛠️ Setup Instructions

### 1️⃣ Install Dependencies

```bash
pip install tensorflow flask librosa numpy scikit-learn
```

### 2️⃣ Train the Model (if not already trained)

```bash
python train_model.py
```

### 3️⃣ Start Flask Server

```bash
python app.py
```

---

## 📦 Files Included

| File | Description |
|------|-------------|
| `train_model.py` | CNN training script |
| `app.py` | Flask API for prediction |
| `audio_model_with_unknown.keras` | Saved trained model |
| `counting dataset/` | Dataset for digits + unknown samples |

---

## ⚠️ Tips & Notes

- **Low Confidence?** If the confidence is below `0.8`, the model treats it as **unknown**.
- Add **more `unknown` samples** for better detection of out-of-vocabulary audio.
- You can extend this model to detect more digits or keywords by modifying:
  - `LABELS` list
  - Dataset folders
  - Retrain the model

---

## 👨‍💻 Author

Developed as part of a Deep Learning project for real-time spoken word classification using TensorFlow and Flask.
