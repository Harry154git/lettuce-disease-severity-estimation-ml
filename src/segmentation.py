from pathlib import Path
import cv2
import numpy as np
import yaml
from tqdm import tqdm


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def segment_leaf_and_lesion(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Mask daun hijau
    lower_green = np.array([25, 25, 25])
    upper_green = np.array([95, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Bersihkan mask daun
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

    # Kandidat bercak: area non-hijau di dalam daun
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lesion_candidates = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(leaf_mask))

    # Threshold untuk bercak
    _, lesion_mask = cv2.threshold(lesion_candidates, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)

    return leaf_mask, lesion_mask


def apply_segmentation(image_rgb):
    leaf_mask, lesion_mask = segment_leaf_and_lesion(image_rgb)

    leaf_only = cv2.bitwise_and(image_rgb, image_rgb, mask=leaf_mask)
    lesion_only = cv2.bitwise_and(image_rgb, image_rgb, mask=lesion_mask)

    return leaf_mask, lesion_mask, leaf_only, lesion_only


def run_segmentation(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    segmented_dir = Path(cfg["paths"]["segmented_dir"])

    classes = [d for d in processed_dir.iterdir() if d.is_dir()]
    if not classes:
        raise FileNotFoundError(f"Tidak ada folder di {processed_dir}")

    for class_dir in classes:
        class_name = class_dir.name
        out_class_dir = segmented_dir / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(class_dir.glob(ext))

        for img_path in tqdm(image_files, desc=f"Segmentasi {class_name}"):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            leaf_mask, lesion_mask, leaf_only, lesion_only = apply_segmentation(img_rgb)

            base = img_path.stem
            cv2.imwrite(str(out_class_dir / f"{base}_leaf_mask.png"), leaf_mask)
            cv2.imwrite(str(out_class_dir / f"{base}_lesion_mask.png"), lesion_mask)
            cv2.imwrite(str(out_class_dir / f"{base}_leaf_only.png"), cv2.cvtColor(leaf_only, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_class_dir / f"{base}_lesion_only.png"), cv2.cvtColor(lesion_only, cv2.COLOR_RGB2BGR))

    print("Segmentasi selesai.")


if __name__ == "__main__":
    run_segmentation()