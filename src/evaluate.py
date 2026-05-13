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

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def plot_confusion_matrix(cm, classes, output_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", 
                     color="white" if cm[i, j] > cm.max() / 2. else "black")

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def evaluate_all_models():
    cfg = load_config()
    base_models_dir = Path(cfg["paths"]["models_dir"])
    features_file = cfg["paths"]["features_file"]
    base_output_dir = Path(cfg["paths"]["cm_dir"])
    metrics_csv_path = Path(cfg["paths"]["results_dir"]) / "metrics.csv"

    if not metrics_csv_path.exists():
        print("[-] Belum ada history training (metrics.csv tidak ditemukan).")
        return

    metrics_df = pd.read_csv(metrics_csv_path)
    df = pd.read_csv(features_file)

    print("\n" + "="*70)
    print(f"{'EVALUASI SEMUA MODEL & VERSI':^70}")
    print("="*70)

    eval_results = []

    for index, row in metrics_df.iterrows():
        version = row.get("version", "v1")
        exp_name = row["experiment"]
        model_name = row["model"]
        test_size = row["test_size"]
        random_state = row["random_state"]
        train_cv_acc = row.get("train_cv_accuracy", None)

        # Cari model di dalam folder versinya
        model_path = base_models_dir / version / f"{model_name}_{exp_name}.pkl"
        le_path = base_models_dir / version / "label_encoder.pkl"
        
        # Fallback jika label encoder tidak ada di folder versi, ambil di root model
        if not le_path.exists():
            le_path = base_models_dir / "label_encoder.pkl"

        if not model_path.exists():
            print(f"[-] Lewati: {model_path} tidak ditemukan.")
            continue

        print(f"\n[+] Evaluasi: {model_name.upper()} | Exp: {exp_name} | Versi: {version}")

        label_encoder = joblib.load(le_path)
        X = df.drop(columns=["label", "image_id"])
        y = label_encoder.transform(df["label"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred)
        
        # Simpan CM terstruktur per versi (contoh: results/confusion_matrix/v1/exp_1/knn_cm.png)
        cm_out_dir = base_output_dir / version / exp_name
        cm_path = cm_out_dir / f"{model_name}_cm.png"
        plot_confusion_matrix(cm, classes=label_encoder.classes_, output_path=cm_path)

        eval_results.append({
            "Version": version,
            "Experiment": exp_name,
            "Model": model_name.upper(),
            "Train_CV_Acc": f"{train_cv_acc*100:.2f}%" if pd.notna(train_cv_acc) else "N/A",
            "Test_Acc": f"{test_acc*100:.2f}%",
            "F1_Score": f"{f1*100:.2f}%"
        })

    print("\n\n" + "="*70)
    print(f"{'TABEL PERBANDINGAN AKURASI TRAINING VS TESTING':^70}")
    print("="*70)
    comparison_df = pd.DataFrame(eval_results)
    print(comparison_df.to_string(index=False))
    print("="*70)
    
    eval_summary_path = Path(cfg["paths"]["results_dir"]) / "evaluation_summary.csv"
    comparison_df.to_csv(eval_summary_path, index=False)
    print(f"\nRingkasan evaluasi komplit tersimpan di: {eval_summary_path}")

if __name__ == "__main__":
    evaluate_all_models()