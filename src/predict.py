import sys
import joblib
import numpy as np
import librosa

MODEL_PATH = "models/pokemon_cry_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

N_MFCC = 13
MAX_LEN = 80
TARGET_SR = 16000


def extract_feature(audio_path):
    y, sr = librosa.load(audio_path, sr=TARGET_SR)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc.reshape(1, -1)


def main():
    if len(sys.argv) < 2:
        print("Usage: py src\\predict.py <audio_path>")
        return

    audio_path = sys.argv[1]

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    X = extract_feature(audio_path)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    pokemon_name = encoder.inverse_transform([prediction])[0]
    confidence = probabilities[prediction]

    print("Predicted Pokémon:", pokemon_name)
    print("Confidence:", round(confidence * 100, 2), "%")


if __name__ == "__main__":
    main()