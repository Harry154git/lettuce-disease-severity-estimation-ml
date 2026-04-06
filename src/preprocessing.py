import os
from pathlib import Path
import cv2
import yaml
from tqdm import tqdm


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_image(image_path, target_size=256, blur_kernel=5):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    return img


def save_image(img_rgb, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_bgr)


def run_preprocessing(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    target_size = cfg["image"]["target_size"]
    blur_kernel = cfg["image"]["blur_kernel"]

    classes = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not classes:
        raise FileNotFoundError(f"Tidak ada folder kelas di {raw_dir}")

    for class_dir in classes:
        class_name = class_dir.name
        out_class_dir = processed_dir / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(class_dir.glob(ext))

        for img_path in tqdm(image_files, desc=f"Preprocessing {class_name}"):
            processed = preprocess_image(img_path, target_size, blur_kernel)
            if processed is None:
                continue
            out_path = out_class_dir / img_path.name
            save_image(processed, out_path)

    print("Preprocessing selesai.")


if __name__ == "__main__":
    run_preprocessing()