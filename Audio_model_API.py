from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io
import librosa

app = Flask(__name__)

# Configuration
label_names = ['eight', 'nine', 'six', 'two', 'zero', 'unknown']
model = tf.keras.models.load_model("audio_model_with_unknown.keras")
CONFIDENCE_THRESHOLD = 0.75
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_audio(waveform, target_length=16000):
    waveform = librosa.util.normalize(waveform)
    if len(waveform) > target_length:
        waveform = waveform[:target_length]
    elif len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)), 'constant')
    return waveform


def get_mel_spectrogram(waveform, sr=16000, n_mels=64):
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = tf.convert_to_tensor(mel_db, dtype=tf.float32)
    mel_db = mel_db[..., tf.newaxis]  # (n_mels, time) → (n_mels, time, 1)
    return mel_db


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to the Audio Classification API",
        "usage": "POST to /predict with audio file in 'file' key",
        "expected_classes": label_names,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        audio_io = io.BytesIO(file.read())
        waveform, sr = librosa.load(audio_io, sr=16000, mono=True)
        waveform = preprocess_audio(waveform)

        mel_spec = get_mel_spectrogram(waveform)  # shape: (64, time, 1)
        mel_spec = tf.expand_dims(mel_spec, axis=0)  # shape: (1, 64, time, 1)

        logits = model.predict(mel_spec, verbose=0)[0]
        probabilities = tf.nn.softmax(logits).numpy()

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        predicted_label = label_names[predicted_index]

        if predicted_label == 'unknown' or confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                "predicted_class": "unknown",
                "confidence": confidence
            }), 200

        return jsonify({
            "predicted_class": predicted_label,
            "confidence": confidence,
            "all_predictions": {label: float(prob) for label, prob in zip(label_names, probabilities)}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
