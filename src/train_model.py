import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


FEATURES_PATH = "models/features.npy"
LABELS_PATH = "models/labels.npy"
MODEL_PATH = "models/pokemon_cry_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


def main():
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)

    print("Loaded features:", X.shape)
    print("Loaded labels:", y.shape)

    # Convert 13 x 80 MFCC matrix into a flat vector
    X = X.reshape(X.shape[0], -1)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", acc)
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=encoder.classes_
        )
    )

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print("\nModel saved:")
    print(MODEL_PATH)
    print(ENCODER_PATH)


if __name__ == "__main__":
    main()