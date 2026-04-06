import argparse
from src.preprocessing import run_preprocessing
from src.segmentation import run_segmentation
from src.feature_extraction import build_feature_csv
from src.train_model import run_training


def main():
    parser = argparse.ArgumentParser(description="Lettuce Disease Severity ML Pipeline")
    parser.add_argument("--step", type=str, required=True,
                        choices=["preprocess", "segment", "features", "train", "all"])
    args = parser.parse_args()

    if args.step == "preprocess":
        run_preprocessing()
    elif args.step == "segment":
        run_segmentation()
    elif args.step == "features":
        build_feature_csv()
    elif args.step == "train":
        run_training()
    elif args.step == "all":
        run_preprocessing()
        run_segmentation()
        build_feature_csv()
        run_training()


if __name__ == "__main__":
    main()