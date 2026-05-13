from pathlib import Path
import json
import joblib
import yaml
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_dataset(features_file):
    df = pd.read_csv(features_file)
    return df

def prepare_xy(df):
    drop_cols = ["label", "image_id"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le

def get_models_and_params(cfg):
    models = {
        "svm": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True))
            ]),
            {
                "clf__C": cfg["models"]["svm"]["C"],
                "clf__kernel": cfg["models"]["svm"]["kernel"],
                "clf__gamma": cfg["models"]["svm"]["gamma"],
            },
        ),
        "knn": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier())
            ]),
            {
                "clf__n_neighbors": cfg["models"]["knn"]["n_neighbors"],
                "clf__weights": cfg["models"]["knn"]["weights"],
                "clf__metric": cfg["models"]["knn"]["metric"],
            },
        ),
    }
    return models

def train_one_model(X_train, y_train, model_name, estimator, param_grid):
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    return grid

def run_training(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    version = cfg.get("version", "v1") # Ambil versi dari config
    
    features_file = cfg["paths"]["features_file"]
    
    # Buat folder per versi di dalam folder models (contoh: models/v1/)
    models_dir = Path(cfg["paths"]["models_dir"]) / version
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(features_file)
    X, y, label_encoder = prepare_xy(df)

    results = []
    model_defs = get_models_and_params(cfg)

    # Simpan label encoder di dalam folder versi masing-masing
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")

    for exp in cfg["experiments"]:
        print(f"\n=== {exp['name']} (Versi: {version}) ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=exp["test_size"],
            random_state=exp["random_state"],
            stratify=y
        )

        for model_name, (estimator, param_grid) in model_defs.items():
            print(f"\nMelatih model: {model_name.upper()} (v: {version})")
            grid = train_one_model(X_train, y_train, model_name, estimator, param_grid)
            best_model = grid.best_estimator_
            
            train_cv_acc = grid.best_score_ 

            y_pred = best_model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)

            # Simpan model di dalam folder versi (contoh: models/v1/knn_exp_1.pkl)
            model_save_path = models_dir / f"{model_name}_{exp['name']}.pkl"
            joblib.dump(best_model, model_save_path)

            results.append({
                "version": version,
                "experiment": exp["name"],
                "model": model_name,
                "test_size": exp["test_size"],
                "random_state": exp["random_state"],
                "best_params": json.dumps(grid.best_params_),
                "train_cv_accuracy": train_cv_acc,
                "test_accuracy": test_acc
            })

            print(f"Best params: {grid.best_params_}")
            print(f"Train CV Accuracy: {train_cv_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")

    # Simpan hasil metrics ke log utama (di-append)
    results_df = pd.DataFrame(results)
    metrics_path = Path(cfg["paths"]["results_dir"]) / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    if metrics_path.exists():
        past_df = pd.read_csv(metrics_path)
        final_df = pd.concat([past_df, results_df], ignore_index=True)
    else:
        final_df = results_df
        
    final_df.to_csv(metrics_path, index=False)
    print(f"\nTraining selesai. Model tersimpan di folder {models_dir}")
    print(f"Metrics ditambahkan ke {metrics_path}")
    return final_df

if __name__ == "__main__":
    run_training()