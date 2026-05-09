import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


FEATURES_PATH = "models/features.npy"
LABELS_PATH = "models/labels.npy"
MODEL_PATH = "models/pokemon_cry_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

OUTPUT_DIR = "evaluation_results"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)

    X = X.reshape(X.shape[0], -1)

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    y_encoded = encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded
    )

    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    max_confidences = np.max(probabilities, axis=1)

    # =========================
    # Top-K Accuracy
    # =========================

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    for i in range(len(y_test)):
        probs = probabilities[i]

        top1 = np.argsort(probs)[-1:]
        top3 = np.argsort(probs)[-3:]
        top5 = np.argsort(probs)[-5:]

        true_label = y_test[i]

        if true_label in top1:
            top1_correct += 1

        if true_label in top3:
            top3_correct += 1

        if true_label in top5:
            top5_correct += 1

    top1_accuracy = top1_correct / len(y_test) * 100
    top3_accuracy = top3_correct / len(y_test) * 100
    top5_accuracy = top5_correct / len(y_test) * 100
    
    target_names = encoder.classes_

    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        os.path.join(OUTPUT_DIR, "classification_report.csv"),
        encoding="utf-8-sig"
    )

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=target_names,
        columns=target_names
    )
    cm_df.to_csv(
        os.path.join(OUTPUT_DIR, "confusion_matrix.csv"),
        encoding="utf-8-sig"
    )

    sample_rows = []

    for i in range(len(y_test)):
        true_label = encoder.inverse_transform([y_test[i]])[0]
        pred_label = encoder.inverse_transform([y_pred[i]])[0]
        confidence = max_confidences[i] * 100

        sample_rows.append({
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": confidence,
            "is_correct": true_label == pred_label
        })

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(
        os.path.join(OUTPUT_DIR, "sample_predictions.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    weak_df = report_df[
        report_df.index.isin(target_names)
    ].sort_values(
        by="recall",
        ascending=True
    )

    weak_df.to_csv(
        os.path.join(OUTPUT_DIR, "weak_classes_by_recall.csv"),
        encoding="utf-8-sig"
    )

    low_conf_df = sample_df.sort_values(
        by="confidence",
        ascending=True
    )

    low_conf_df.to_csv(
        os.path.join(OUTPUT_DIR, "low_confidence_predictions.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("Evaluation completed.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nWorst 20 classes by recall:")
    print(weak_df[["precision", "recall", "f1-score", "support"]].head(20))

    print("\nLowest 20 confidence predictions:")
    print(low_conf_df.head(20))

    print("\n========================")
    print("Top-K Accuracy")
    print("========================")

    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

if __name__ == "__main__":
    main()