from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import yaml
from tqdm import tqdm


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_color_features(img_rgb):
    feats = {}
    channels = cv2.split(img_rgb)

    names = ["r", "g", "b"]
    for ch, name in zip(channels, names):
        feats[f"{name}_mean"] = float(np.mean(ch))
        feats[f"{name}_std"] = float(np.std(ch))
        feats[f"{name}_min"] = float(np.min(ch))
        feats[f"{name}_max"] = float(np.max(ch))

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    feats["h_mean"] = float(np.mean(h))
    feats["s_mean"] = float(np.mean(s))
    feats["v_mean"] = float(np.mean(v))

    return feats


def extract_texture_features(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    glcm = graycomatrix(
        gray,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    feats = {
        "contrast": float(graycoprops(glcm, "contrast").mean()),
        "dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
        "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "energy": float(graycoprops(glcm, "energy").mean()),
        "correlation": float(graycoprops(glcm, "correlation").mean()),
    }
    return feats


def extract_area_features(leaf_mask, lesion_mask):
    leaf_pixels = np.count_nonzero(leaf_mask)
    lesion_pixels = np.count_nonzero(lesion_mask)

    lesion_ratio = lesion_pixels / leaf_pixels if leaf_pixels > 0 else 0.0

    return {
        "leaf_pixels": int(leaf_pixels),
        "lesion_pixels": int(lesion_pixels),
        "lesion_ratio": float(lesion_ratio),
    }


def extract_features_from_image(img_rgb, leaf_mask, lesion_mask):
    features = {}
    features.update(extract_color_features(img_rgb))
    features.update(extract_texture_features(img_rgb))
    features.update(extract_area_features(leaf_mask, lesion_mask))
    return features


def build_feature_csv(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    segmented_dir = Path(cfg["paths"]["segmented_dir"])
    features_file = Path(cfg["paths"]["features_file"])
    features_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    class_dirs = [d for d in segmented_dir.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        label = class_dir.name

        # Ambil semua leaf_only (lebih stabil)
        leaf_images = list(class_dir.glob("*_leaf_only.png"))

        for leaf_path in tqdm(leaf_images, desc=f"Ekstraksi fitur {label}"):

            base = leaf_path.name.replace("_leaf_only.png", "")

            leaf_mask_path = class_dir / f"{base}_leaf_mask.png"
            lesion_mask_path = class_dir / f"{base}_lesion_mask.png"

            if not leaf_mask_path.exists() or not lesion_mask_path.exists():
                continue

            # Load data
            img_bgr = cv2.imread(str(leaf_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            leaf_mask = cv2.imread(str(leaf_mask_path), cv2.IMREAD_GRAYSCALE)
            lesion_mask = cv2.imread(str(lesion_mask_path), cv2.IMREAD_GRAYSCALE)

            # Extract features
            feats = extract_features_from_image(img_rgb, leaf_mask, lesion_mask)
            feats["label"] = label
            feats["image_id"] = base

            rows.append(feats)

    if len(rows) == 0:
        raise ValueError("Tidak ada fitur yang berhasil diekstrak. Cek pipeline segmentasi!")

    df = pd.DataFrame(rows)
    df.to_csv(features_file, index=False)

    print(f"Fitur berhasil dibuat: {len(df)} data")
    return df


if __name__ == "__main__":
    build_feature_csv()