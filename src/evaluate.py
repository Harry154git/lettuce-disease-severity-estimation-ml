import json
from pathlib import Path

import joblib
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ===============================
# LOAD CONFIG
# ===============================
def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===============================
# PLOT CONFUSION MATRIX
# ===============================
def plot_confusion_matrix(cm, classes, output_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # isi angka di matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center")

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ===============================
# EVALUATE MODEL (NO DATA LEAKAGE)
# ===============================
def evaluate_model(model_path, features_file, label_encoder, output_dir, cfg):
    df = pd.read_csv(features_file)

    # Pisahkan fitur & label
    X = df.drop(columns=["label", "image_id"])
    y = label_encoder.transform(df["label"])

    # Ambil parameter split dari config
    test_size = cfg["split"]["test_size"]
    random_state = cfg["split"]["random_state"]
    stratify_flag = cfg["split"]["stratify"]

    stratify_y = y if stratify_flag else None

    # SPLIT DATA (INI YANG SEBELUMNYA SALAH)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y
    )

    # Load model
    model = joblib.load(model_path)

    # Predict ONLY test data
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix image
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, classes=label_encoder.classes_, output_path=cm_path)

    # Save metrics
    metrics = {
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1)
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Print hasil
    print("\n=== Evaluation (NO DATA LEAKAGE) ===")
    print(json.dumps(metrics, indent=2))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    return metrics


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    cfg = load_config()

    models_dir = Path(cfg["paths"]["models_dir"])
    features_file = cfg["paths"]["features_file"]
    output_dir = cfg["paths"]["cm_dir"]

    # Load label encoder
    label_encoder = joblib.load(models_dir / "label_encoder.pkl")

    # 👉 PILIH MODEL TERBAIK (ubah sesuai hasil metrics.csv)
    model_path = models_dir / "svm_exp_2.pkl"

    evaluate_model(
        model_path=model_path,
        features_file=features_file,
        label_encoder=label_encoder,
        output_dir=output_dir,
        cfg=cfg
    )