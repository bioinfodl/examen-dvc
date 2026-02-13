import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model using best params.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed_data"),
        help="Directory containing processed datasets.",
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=Path("models/best_params.pkl"),
        help="Path to best parameters file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/model.joblib"),
        help="Path to save trained model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(args.processed_dir / "X_train_scaled.csv")
    y_train = pd.read_csv(args.processed_dir / "y_train.csv").squeeze("columns")

    best_params = joblib.load(args.params_path)
    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)

    joblib.dump(model, args.model_path)


if __name__ == "__main__":
    main()
